import json
import os
import copy
import torch
import argparse
from tqdm import tqdm
from xtuner.registry import BUILDER
from mmengine.config import Config
from xtuner.model.utils import guess_load_checkpoint
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from einops import rearrange


class WISE(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = copy.deepcopy(self.data[idx])
        return data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--data', default='data/wise/cultural_common_sense.json', type=str)
    parser.add_argument('--output', default='output', type=str)
    parser.add_argument("--cfg_prompt", type=str, default=None)
    parser.add_argument("--cfg_scale", type=float, default=4.5)
    parser.add_argument('--num_steps', type=int, default=20)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    accelerator = Accelerator()
    # each GPU creates a string
    message = [f"Hello this is GPU {accelerator.process_index}"]
    # collect the messages from all GPUs
    messages = gather_object(message)
    # output the messages only on the main process with accelerator.print()
    accelerator.print(f"Number of gpus: {accelerator.num_processes}")
    accelerator.print(messages)

    config = Config.fromfile(args.config)

    print(f'Device: {accelerator.device}', flush=True)

    dataset = WISE(data_path=args.data)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            collate_fn=lambda x: x
                            )

    model = BUILDER.build(config.model)
    if args.checkpoint is not None:
        state_dict = guess_load_checkpoint(args.checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        accelerator.print(f"Unexpected parameters: {unexpected}")
    model = model.to(device=accelerator.device)
    model = model.to(model.dtype)
    model.eval()

    dataloader = accelerator.prepare(dataloader)

    print(f'Number of samples: {len(dataloader)}', flush=True)

    if args.cfg_prompt is None:
        cfg_prompt = model.prompt_template['CFG']
    else:
        cfg_prompt = model.prompt_template['GENERATION'].format(input=args.cfg_prompt.strip())
    cfg_prompt = model.prompt_template['INSTRUCTION'].format(input=cfg_prompt)
    cfg_prompt += model.prompt_template['IMG_START_TOKEN']

    if accelerator.is_main_process:
        os.makedirs(args.output, exist_ok=True)

    generator = torch.Generator(device=model.device).manual_seed(args.seed)

    for batch_idx, data_samples in tqdm(enumerate(dataloader), disable=not accelerator.is_main_process):
        device_idx = accelerator.process_index

        prompts = []
        for data_sample in data_samples:
            prompt = copy.deepcopy(data_sample['Prompt'].strip())
            prompt = model.prompt_template['GENERATION'].format(input=prompt)
            prompt = model.prompt_template['INSTRUCTION'].format(input=prompt)
            prompt += model.prompt_template['IMG_START_TOKEN']
            prompts.append(prompt)

        # prompts = prompts * 4
        prompts = prompts + len(prompts) * [cfg_prompt]

        inputs = model.tokenizer(
            prompts, add_special_tokens=True, return_tensors='pt', padding=True).to(accelerator.device)

        images = model.generate(**inputs, progress_bar=False,
                                cfg_scale=args.cfg_scale, num_steps=args.num_steps,
                                generator=generator, height=args.height, width=args.width)
        images = rearrange(images, 'b c h w -> b h w c')

        images = torch.clamp(
            127.5 * images + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for image, data_sample in zip(images, data_samples):
            prompt_id = data_sample['prompt_id']
            Image.fromarray(image).save(f"{args.output}/{prompt_id}.png")
