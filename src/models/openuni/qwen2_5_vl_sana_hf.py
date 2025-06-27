import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T
from xtuner.registry import BUILDER
from mmengine.model import BaseModel
from mmengine.logging import print_log
from torch.nn.utils.rnn import pad_sequence
from xtuner.model.utils import guess_load_checkpoint
from diffusers.pipelines.sana.pipeline_sana import SanaPipeline
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from peft import LoraConfig
from src.models.connector import ConnectorConfig, ConnectorEncoder
from einops import rearrange, reduce


IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            names = name.split('.')
            lora_module = names[0] if len(names) == 1 else names[-1]
            if lora_module == '0':
                lora_module = 'to_out.0'
            lora_module_names.add(lora_module)

    return list(lora_module_names)


class OpenUniQwen25VLSANAHF(BaseModel):
    def __init__(self,
                 lmm,
                 transformer,
                 train_scheduler,
                 test_scheduler,
                 vae,
                 tokenizer,
                 prompt_template,
                 connector,
                 num_queries=256,
                 pretrained_pth=None,
                 use_activation_checkpointing=True,
                 lora_modules=None,  # ["to_k", "to_q", "to_v", 'to_out.0'],
                 lora_rank=8,
                 lora_alpha=8,
                 freeze_lmm=True,
                 freeze_transformer=True,
                 vit_input_size=448,
                 max_length=2048,
                 proj_type='enc_proj'
                 ):
        super().__init__()
        self.use_activation_checkpointing = use_activation_checkpointing

        self.lmm = BUILDER.build(lmm)

        if freeze_lmm:
            self.lmm.requires_grad_(False)
        self.freeze_lmm = freeze_lmm

        self.train_scheduler = BUILDER.build(train_scheduler)
        self.test_scheduler = BUILDER.build(test_scheduler)

        self.transformer = BUILDER.build(transformer)
        if freeze_transformer:
            self.transformer.requires_grad_(False)
        self.freeze_transformer = freeze_transformer
        if lora_modules is not None:
            if lora_modules == 'auto':
                lora_modules = find_all_linear_names(self.transformer)

            transformer_lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights="gaussian",
                target_modules=lora_modules,
            )
            self.transformer.add_adapter(transformer_lora_config)


        self.vae = BUILDER.build(vae)
        self.vae.requires_grad_(False)

        self.tokenizer = BUILDER.build(tokenizer)
        self.prompt_template = prompt_template
        self.vit_input_size = vit_input_size
        self.max_length = max_length
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(prompt_template['IMG_CONTEXT_TOKEN'])
        self.register_buffer('vit_mean', torch.tensor(IMAGE_MEAN), persistent=False)
        self.register_buffer('vit_std', torch.tensor(IMAGE_STD), persistent=False)

        self.num_queries = num_queries
        self.connector = ConnectorEncoder(ConnectorConfig(**connector))

        self.proj_type = proj_type
        if self.proj_type == 'proj_enc':
            assert self.connector.config.hidden_size == self.transformer.config.caption_channels
            self.projector = nn.Linear(
                self.llm.config.hidden_size, self.connector.config.hidden_size)
        elif self.proj_type == 'enc_proj':
            assert self.connector.config.hidden_size == self.llm.config.hidden_size
            self.projector = nn.Linear(
                self.connector.config.hidden_size, self.transformer.config.caption_channels)
        elif self.proj_type == 'proj_enc_proj':
            self.projector = nn.ModuleList([
                nn.Linear(self.llm.config.hidden_size, self.connector.config.hidden_size),
                nn.Linear(self.connector.config.hidden_size, self.transformer.config.caption_channels)
            ])
        else:
            raise ValueError(f'Unknown proj type: {self.proj_type}')

        self.meta_queries = nn.Parameter(
            torch.zeros(num_queries, self.llm.config.hidden_size))
        nn.init.normal_(self.meta_queries, std=1 / math.sqrt(self.llm.config.hidden_size))

        if use_activation_checkpointing:
            self.llm.enable_input_require_grads()
            self.gradient_checkpointing_enable()

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            info = self.load_state_dict(pretrained_state_dict, strict=False)
            print_log(f'Load pretrained weight from {pretrained_pth}')

    def llm2dit(self, x):
        if self.proj_type == 'proj_enc':
            return self.connector(self.projector(x))
        elif self.proj_type == 'enc_proj':
            return self.projector(self.connector(x))
        elif self.proj_type == 'proj_enc_proj':
            return self.projector[1](self.connector(self.projector[0](x)))
        else:
            raise ValueError(f'Unknown proj type: {self.proj_type}')

    @property
    def llm(self):
        return self.lmm.language_model

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        self.transformer.enable_gradient_checkpointing()
        self.connector.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.transformer.disable_gradient_checkpointing()
        self.connector.gradient_checkpointing = False

    @property
    def device(self):
        return self.llm.device

    @property
    def dtype(self):
        return self.llm.dtype

    def train(self: T, mode: bool = True) -> T:
        super().train(mode=mode)
        if self.vae is not None:
            self.vae.train(mode=False)
        if not mode:
            self.gradient_checkpointing_disable()

        return self

    @torch.no_grad()
    def pixels_to_latents(self, x):
        scaling_factor = self.vae.config.scaling_factor
        z = self.vae.encode(x)[0] * scaling_factor
        return z

    @torch.no_grad()
    def latents_to_pixels(self, z):
        scaling_factor = self.vae.config.scaling_factor
        x_rec = self.vae.decode(z / scaling_factor)[0]
        return x_rec

    def prepare_forward_input(self,
                              x,
                              input_ids=None,
                              image_embeds=None,
                              image_grid_thw=None,
                              attention_mask=None,
                              past_key_values=None):
        b, l, _ = x.shape
        assert l > 0
        attention_mask = attention_mask.to(device=self.device, dtype=torch.bool)

        assert l == self.num_queries

        input_ids = torch.cat([input_ids, input_ids.new_zeros(b, l)], dim=1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(b, l)], dim=1)

        position_ids, _ = self.lmm.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            second_per_grid_ts=None,
            attention_mask=attention_mask,
        )

        # prepare context
        if past_key_values is not None:
            inputs_embeds = x
            position_ids = position_ids[..., -l:]
        else:
            input_ids = input_ids[:, :-l]   # context input_ids

            if image_embeds is None:
                inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            else:
                inputs_embeds = torch.zeros(*input_ids.shape, self.llm.config.hidden_size,
                                            device=self.device, dtype=self.dtype)
                inputs_embeds[input_ids == self.image_token_id] = image_embeds.flatten(0, 1)
                inputs_embeds[input_ids != self.image_token_id] = self.llm.get_input_embeddings()(
                    input_ids[input_ids != self.image_token_id]
                )

            inputs_embeds = torch.cat([inputs_embeds, x], dim=1)

        inputs = dict(inputs_embeds=inputs_embeds,
                      attention_mask=attention_mask,
                      position_ids=position_ids,
                      past_key_values=past_key_values)


        return inputs

    def forward(self, data, data_samples=None, mode='loss'):
        if mode == 'loss':
            return self.compute_loss(data_dict=data)
        else:
            raise NotImplementedError

    def compute_loss(self, data_dict):
        losses = {}
        for data_type in ['text2image', 'image2image']:
            if data_type in data_dict:
                losses[f'loss_{data_type}'] = getattr(self, f'{data_type}_loss')(data_dict[data_type])
        if len(losses) == 0:
            if 'pixel_values_src' in data_dict:
                losses[f'loss_image2image'] = self.image2image_loss(data_dict)
            else:
                losses[f'loss_text2image'] = self.text2image_loss(data_dict)

        return losses

    @torch.no_grad()
    def get_semantic_features(self, pixel_values):
        # pixel_values: [-1, 1]
        pixel_values = (pixel_values + 1.0) / 2     # [0, 1]
        pixel_values = pixel_values - self.vit_mean.view(1, 3, 1, 1)
        pixel_values = pixel_values / self.vit_std.view(1, 3, 1, 1)

        pixel_values = F.interpolate(pixel_values, size=(self.vit_input_size, self.vit_input_size),
                                     mode='bilinear')
        b, c, h, w = pixel_values.shape

        patch_size = self.lmm.config.vision_config.patch_size
        spatial_merge_size = self.lmm.config.vision_config.spatial_merge_size
        temporal_patch_size = self.lmm.config.vision_config.temporal_patch_size

        pixel_values = pixel_values[:, None].expand(b, temporal_patch_size, c, h, w)

        grid_t = 1
        grid_h, grid_w = h // patch_size, w // patch_size

        pixel_values = pixel_values.view(
            b,
            grid_t,
            temporal_patch_size,
            c,
            grid_h // spatial_merge_size,
            spatial_merge_size,
            patch_size,
            grid_w // spatial_merge_size,
            spatial_merge_size,
            patch_size,
        )

        pixel_values = rearrange(
            pixel_values, 'b t tp c h m p w n q -> (b t h w m n) (c tp p q)')

        image_grid_thw = torch.tensor([(grid_t, grid_h, grid_w)] * b).to(self.device).long()

        image_embeds = self.lmm.visual(pixel_values, grid_thw=image_grid_thw)
        image_embeds = rearrange(image_embeds, '(b l) d -> b l d', b=b)

        return image_embeds, image_grid_thw


    @torch.no_grad()
    def prepare_text_conditions(self, prompt, cfg_prompt=None):
        if cfg_prompt is None:
            cfg_prompt = self.prompt_template['CFG']
        else:
            cfg_prompt = self.prompt_template['GENERATION'].format(input=cfg_prompt.strip())
        prompt = self.prompt_template['GENERATION'].format(input=prompt.strip())

        all_prompts = [
            self.prompt_template['INSTRUCTION'].format(input=prompt),
            self.prompt_template['INSTRUCTION'].format(input=cfg_prompt),
        ]

        input_ids = [self.tokenizer.encode(p, add_special_tokens=True, return_tensors='pt')[0]
                     for p in all_prompts]
        valid_lens = [len(input_ids_) for input_ids_ in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True,
                                 padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.zeros_like(input_ids).bool()
        for i in range(len(input_ids)):
            attention_mask[i, :valid_lens[i]] = True

        return dict(input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device))

    def text2image_loss(self, data_dict):

        # obtain image latents
        if 'image_latents' in data_dict:
            image_latents = data_dict['image_latents'].to(dtype=self.dtype, device=self.device)
        else:
            pixel_values = data_dict['pixel_values'].to(dtype=self.dtype, device=self.device)
            image_latents = self.pixels_to_latents(pixel_values)

        b, _, height, weight = image_latents.shape

        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)
        hidden_states = self.meta_queries[None].expand(b, self.num_queries, -1)

        inputs = self.prepare_forward_input(x=hidden_states,
                                            input_ids=input_ids,
                                            attention_mask=attention_mask)

        output = self.llm(**inputs, return_dict=True)
        hidden_states = output.last_hidden_state[:, -self.num_queries:]
        hidden_states = self.llm2dit(hidden_states)

        loss_diff = self.diff_loss(model_input=image_latents,
                                   prompt_embeds=hidden_states,
                                   prompt_attention_mask=None)

        return loss_diff

    def image2image_loss(self, data_dict):

        pixel_values_src = data_dict['pixel_values_src'].to(dtype=self.dtype, device=self.device)
        image_embeds, image_grid_thw = self.get_semantic_features(pixel_values_src)

        pixel_values = data_dict['pixel_values'].to(dtype=self.dtype, device=self.device)
        image_latents = self.pixels_to_latents(pixel_values)

        b, _, height, weight = image_latents.shape

        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)

        hidden_states = self.meta_queries[None].expand(b, self.num_queries, -1)

        inputs = self.prepare_forward_input(x=hidden_states,
                                            input_ids=input_ids,
                                            image_embeds=image_embeds,
                                            image_grid_thw=image_grid_thw,
                                            attention_mask=attention_mask)

        inputs_embeds = inputs['inputs_embeds']
        attention_mask = inputs['attention_mask']
        position_ids = inputs['position_ids']

        max_length = self.max_length
        if inputs_embeds.shape[1] > max_length:
            inputs_embeds = inputs_embeds[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]
            position_ids = position_ids[..., -max_length:]

        output = self.llm(inputs_embeds=inputs_embeds,
                          attention_mask=attention_mask,
                          position_ids=position_ids,
                          return_dict=True)

        hidden_states = output.last_hidden_state[:, -self.num_queries:]
        hidden_states = self.llm2dit(hidden_states)

        loss_diff = self.diff_loss(model_input=image_latents,
                                   prompt_embeds=hidden_states,
                                   prompt_attention_mask=None)

        return loss_diff

    @torch.no_grad()
    def generate(self,
                 input_ids=None,
                 attention_mask=None,
                 image_embeds=None,
                 image_grid_thw=None,
                 cfg_scale=4.5,
                 num_steps=20,
                 generator=None,
                 height=512,
                 width=512,
                 progress_bar=True,
                 **kwargs):

        bsz = attention_mask.shape[0]

        assert bsz % 2 == 0

        hidden_states = self.meta_queries[None].expand(bsz, self.num_queries, -1)
        inputs = self.prepare_forward_input(x=hidden_states,
                                            input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            image_embeds=image_embeds,
                                            image_grid_thw=image_grid_thw)

        output = self.llm(**inputs, return_dict=True)
        hidden_states = output.last_hidden_state[:, -self.num_queries:]

        hidden_states = self.llm2dit(hidden_states)
        attention_mask = torch.ones(bsz, self.num_queries, device=self.device, dtype=torch.bool)

        pipeline = SanaPipeline(transformer=self.transformer,
                                scheduler=self.test_scheduler,
                                vae=self.vae, text_encoder=None, tokenizer=None
                                )
        pipeline.set_progress_bar_config(disable=not progress_bar)

        samples = pipeline(
            negative_prompt=None,
            height=height,
            width=width,
            prompt_embeds=hidden_states[:bsz // 2],
            prompt_attention_mask=attention_mask[:bsz // 2],
            negative_prompt_embeds=hidden_states[bsz // 2:],
            negative_prompt_attention_mask=attention_mask[bsz // 2:],
            num_inference_steps=num_steps,
            generator=generator,
            complex_human_instruction=None,
            output_type='latent',
            use_resolution_binning=False,
            guidance_scale=cfg_scale,
        ).images.to(self.dtype)

        return self.latents_to_pixels(samples)

    def diff_loss(self, model_input, prompt_embeds, prompt_attention_mask):
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]

        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme="none",
            batch_size=bsz,
        )
        indices = (u * self.train_scheduler.config.num_train_timesteps).long()
        timesteps = self.train_scheduler.timesteps[indices].to(device=model_input.device)

        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        sigmas = self.get_sigmas(timesteps, n_dim=model_input.ndim)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        # Predict the noise residual
        model_pred = self.transformer(
            hidden_states=noisy_model_input,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            timestep=timesteps,
            return_dict=False,
        )[0]

        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)

        # flow matching loss
        target = noise - model_input

        # Compute regular loss.
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()

        return loss

    def get_sigmas(self, timesteps, n_dim=4):
        sigmas = self.train_scheduler.sigmas.to(device=self.device, dtype=self.dtype)
        schedule_timesteps = self.train_scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma


if __name__ == '__main__':
    import argparse
    import numpy as np
    from PIL import Image
    from mmengine.config import Config
    from xtuner.dataset.utils import expand2square
    from transformers.cache_utils import DynamicCache
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file path.')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--image", type=str, default="figures/view.jpg")
    parser.add_argument("--prompt", type=str, default="Briefly describe the image.")
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--max_new_tokens", type=int, default=128)

    args = parser.parse_args()

    config = Config.fromfile(args.config)
    model = BUILDER.build(config.model).eval().cuda()
    model = model.to(model.dtype)

    image = Image.open(args.image)

    image = expand2square(pil_img=image, background_color=(127, 127, 127))
    image = image.resize(size=(args.height, args.width))
    image = torch.from_numpy(np.array(image)).to(dtype=model.dtype, device=model.device)
    image = rearrange(image, 'h w c -> c h w')[None]
    image = 2 * (image / 255) - 1

    with torch.no_grad():
        image_embeds, image_grid_thw = model.get_semantic_features(image)

    prompt_template = model.prompt_template
    image_tokens = prompt_template['IMG_START_TOKEN'] + \
                   prompt_template['IMG_CONTEXT_TOKEN'] * image_embeds.shape[1] + \
                   prompt_template['IMG_END_TOKEN']

    prompt = f'{image_tokens}\n{args.prompt}'
    prompt = prompt_template['INSTRUCTION'].format(input=prompt)

    inputs = model.tokenizer(
        [prompt, ], add_special_tokens=True, return_tensors='pt', padding=True).to(model.device)

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    position_ids, _ = model.lmm.model.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        second_per_grid_ts=None,
        attention_mask=attention_mask,
    )

    inputs_embeds = torch.zeros(*input_ids.shape, model.llm.config.hidden_size,
                                device=model.device, dtype=model.dtype)
    inputs_embeds[input_ids == model.image_token_id] = image_embeds.flatten(0, 1)
    inputs_embeds[input_ids != model.image_token_id] = model.llm.get_input_embeddings()(
        input_ids[input_ids != model.image_token_id]
    )
    past_key_values = DynamicCache.from_legacy_cache()

    output_ids = []

    b = 1

    for idx in tqdm(range(args.max_new_tokens)):
        import pdb; pdb.set_trace()
        output = model.llm(inputs_embeds=inputs_embeds,
                           attention_mask=attention_mask,
                           position_ids=position_ids,
                           past_key_values=past_key_values,
                           use_cache=True,
                           return_dict=True)

        hidden_states = output.last_hidden_state[:, -1]   # b d
        logits = model.lmm.lm_head(hidden_states)   # b n

        pred_idx = torch.argmax(logits, dim=-1)  # b

        output_ids.append(pred_idx)

        inputs_embeds = model.llm.get_input_embeddings()(pred_idx)[:, None]    # b 1 d
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(b, 1)], dim=1)

        position_ids_max = reduce(position_ids, 'n b l -> b', reduction='max').view(1, b, 1)

        position_ids = position_ids_max.expand(3, b, 1) + 1
        past_key_values = output.past_key_values

    output_ids = torch.stack(output_ids, dim=1)


    output_text = model.tokenizer.decode(output_ids[0])

    print(output_text)

