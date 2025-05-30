import numpy as np
import torch
from PIL import Image
from mmengine.config import Config
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file path.')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--image", type=str, default="data/view.jpg")
    parser.add_argument("--prompt", type=str, default="Keep the image as it is.")
    parser.add_argument("--cfg_prompt", type=str, default="Keep the image as it is.")
    parser.add_argument("--cfg_scale", type=float, default=4.5)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid_size", type=int, default=2)
    parser.add_argument('--output', type=str, default='output.jpg')

    args = parser.parse_args()

    config = Config.fromfile(args.config)
    model = BUILDER.build(config.model).eval().cuda()
    model = model.to(model.dtype)
    checkpoint = guess_load_checkpoint(args.checkpoint)
    info = model.load_state_dict(checkpoint, strict=False)

    image = Image.open(args.image)
    image = image.resize(size=(args.height, args.width))
    image = torch.from_numpy(np.array(image)).to(dtype=model.dtype, device=model.device)
    image = rearrange(image, 'h w c -> c h w')[None]
    image = 2 * (image / 255) - 1

    vit_embeds = model.get_semantic_features(image)

    prompt_template = model.prompt_template
    image_tokens = prompt_template['IMG_START_TOKEN'] + \
                   prompt_template['IMG_CONTEXT_TOKEN'] * vit_embeds.shape[1] + \
                   prompt_template['IMG_END_TOKEN']

    prompt = f'{image_tokens}\n{args.prompt}'
    prompt = prompt_template['INSTRUCTION'].format(input=prompt)
    prompt += prompt_template['IMG_START_TOKEN']

    cfg_prompt = f'{image_tokens}\n{args.cfg_prompt}'
    cfg_prompt = prompt_template['INSTRUCTION'].format(input=cfg_prompt)
    cfg_prompt += prompt_template['IMG_START_TOKEN']

    inputs = model.tokenizer(
        [prompt, cfg_prompt], add_special_tokens=True, return_tensors='pt', padding=True).to(model.device)

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    inputs_embeds = model.llm.get_input_embeddings()(input_ids)
    inputs_embeds[input_ids == model.image_token_id] = vit_embeds.expand(2, -1, -1).flatten(0, 1)

    # repeat
    bsz = args.grid_size ** 2
    inputs_embeds = inputs_embeds[:, None].expand(-1, bsz, -1, -1).flatten(0, 1)
    attention_mask = attention_mask[:, None].expand(-1, bsz, -1).flatten(0, 1)

    generator = torch.Generator(device=model.device).manual_seed(args.seed)
    samples = model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                             cfg_scale=args.cfg_scale, num_steps=args.num_steps,
                             generator=generator, height=args.height, width=args.width)

    samples = rearrange(samples, '(m n) c h w -> (m h) (n w) c', m=args.grid_size, n=args.grid_size)
    samples = torch.clamp(
        127.5 * samples + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()

    Image.fromarray(samples).save(args.output)
