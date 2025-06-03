from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from src.datasets.collate_functions import collate_func_gen
from src.datasets.text2image.blip3_o import BLIP3oDataset


with read_base():
    from .processors import prompt_template, tokenizer, image_size, pad_index

max_length = 128

dataset = dict(type=BLIP3oDataset,
               image_size=image_size,
               data_path='data/BLIP3o-60k/data.json',
               cap_folder='data/BLIP3o-60k/raw',
               image_folder='data/BLIP3o-60k/raw',
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
    collate_fn=dict(type=collate_func_gen,
                    pad_index=pad_index)
)
