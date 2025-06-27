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
from src.datasets.utils import load_jsonl
from einops import rearrange
import inflect
p = inflect.engine()


class GenEval(Dataset):
    def __init__(self, data_path):
        self.data = load_jsonl(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = copy.deepcopy(self.data[idx])
        # if data_dict['tag'] == "position":
        #     obj0 = data_dict['include'][0]["class"]
        #     obj1 = data_dict['include'][1]["class"]
        #     relation = data_dict['include'][1]['position'][0]
        #     if relation == 'left of':
        #         prompt = f'{p.a(obj0)} on the right and {p.a(obj1)} on the left.'
        #     elif relation == 'right of':
        #         prompt = f'{p.a(obj0)} on the left and {p.a(obj1)} on the right.'
        #     elif relation == 'below':
        #         prompt = f'{p.a(obj0)} above and {p.a(obj1)} below.'
        #     elif relation == 'above':
        #         prompt = f'{p.a(obj0)} below and {p.a(obj1)} above.'
        #     else:
        #         raise ValueError
        #     text = prompt
        # else:
        #     text = data_dict['prompt']    # .replace('a photo of ', '')

        text = data_dict['prompt']

        data_dict.update(idx=idx, text=text)

        return data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--data', default='data/geneval/prompts/evaluation_metadata.jsonl', type=str)
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

    dataset = GenEval(data_path=args.data)
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
    if model.prompt_template.get('IMG_START_TOKEN_FOR_GENERATION', True):
        cfg_prompt += model.prompt_template['IMG_START_TOKEN']

    if accelerator.is_main_process:
        os.makedirs(args.output, exist_ok=True)

    generator = torch.Generator(device=model.device).manual_seed(args.seed)

    for batch_idx, data_samples in tqdm(enumerate(dataloader), disable=not accelerator.is_main_process):
        device_idx = accelerator.process_index

        prompts = []
        for data_sample in data_samples:
            prompt = copy.deepcopy(data_sample['text'].strip())
            prompt = model.prompt_template['GENERATION'].format(input=prompt)
            prompt = model.prompt_template['INSTRUCTION'].format(input=prompt)
            if model.prompt_template.get('IMG_START_TOKEN_FOR_GENERATION', True):
                prompt += model.prompt_template['IMG_START_TOKEN']
            prompts.append(prompt)

        prompts = prompts * 4
        prompts = prompts + len(prompts) * [cfg_prompt]

        inputs = model.tokenizer(
            prompts, add_special_tokens=True, return_tensors='pt', padding=True).to(accelerator.device)

        images = model.generate(**inputs, progress_bar=False,
                                cfg_scale=args.cfg_scale, num_steps=args.num_steps,
                                generator=generator, height=args.height, width=args.width)
        images = rearrange(images, '(n b) c h w -> b n h w c', n=4)

        images = torch.clamp(
            127.5 * images + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for image, data_sample in zip(images, data_samples):
            sample_id = data_sample.pop('idx')
            os.makedirs(f"{args.output}/{sample_id:05d}", exist_ok=True)

            with open(f"{args.output}/{sample_id:05d}/metadata.jsonl", "w") as jsonl_file:
                jsonl_file.write(json.dumps(data_sample) + "\n")

            os.makedirs(f"{args.output}/{sample_id:05d}/samples", exist_ok=True)

            for i, sub_image in enumerate(image):
                Image.fromarray(sub_image).save(f"{args.output}/{sample_id:05d}/samples/{i:04d}.png")
