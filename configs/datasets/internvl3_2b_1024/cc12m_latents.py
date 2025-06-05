from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from src.datasets.collate_functions import collate_func_gen_latents
from src.datasets.text2image.caption_datasets import CaptionDataset


with read_base():
    from .processors import prompt_template, tokenizer, image_size, pad_index

max_length = 128

dataset = dict(type=CaptionDataset,
               image_size=image_size,
               cap_folder='data/cc12m/captions',
               data_path='data/cc12m/data.json',
               image_latents_folder=f'data/cc12m/raw_dc32_{image_size}',
               # image_folder='data/cc12m/raw',
               unconditional=0.1,
               prompt_template=prompt_template,
               ceph_folder=None,
               ceph_config=None,
               tokenizer=tokenizer,
               max_length=max_length)

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    collate_fn=dict(type=collate_func_gen_latents,
                    pad_index=pad_index)
)
