import torch
import argparse
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint
from mmengine.config import Config
from einops import rearrange
from PIL import Image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='log file path.')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--prompt", type=str, default='a dog on the left and a cat on the right.')
    parser.add_argument("--cfg_prompt", type=str, default=None)
    parser.add_argument("--cfg_scale", type=float, default=4.5)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid_size", type=int, default=2)
    parser.add_argument('--output', type=str, default='output.jpg')

    args = parser.parse_args()
    config = Config.fromfile(args.config)
    model = BUILDER.build(config.model).cuda().bfloat16().eval()

    if args.checkpoint is not None:
        print(f"Load checkpoint: {args.checkpoint}", flush=True)
        checkpoint = guess_load_checkpoint(args.checkpoint)
        info = model.load_state_dict(checkpoint, strict=False)

    generator = torch.Generator(device=model.device).manual_seed(args.seed)
    class_info = model.prepare_text_conditions(args.prompt, args.cfg_prompt)
    input_ids = class_info['input_ids']
    attention_mask = class_info['attention_mask']

    assert len(input_ids) == 2  # the last one is unconditional prompt

    # repeat
    bsz = args.grid_size ** 2
    input_ids = torch.cat([
        input_ids[:1].expand(bsz, -1),
        input_ids[1:].expand(bsz, -1),
    ])
    attention_mask = torch.cat([
        attention_mask[:1].expand(bsz, -1),
        attention_mask[1:].expand(bsz, -1),
    ])

    samples = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                             cfg_scale=args.cfg_scale, num_steps=args.num_steps,
                             generator=generator, height=args.height, width=args.width)

    samples = rearrange(samples, '(m n) c h w -> (m h) (n w) c', m=args.grid_size, n=args.grid_size)
    samples = torch.clamp(
        127.5 * samples + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()

    Image.fromarray(samples).save(args.output)
