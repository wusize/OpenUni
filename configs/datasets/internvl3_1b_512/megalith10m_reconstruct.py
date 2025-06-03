from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from src.datasets.collate_functions import collate_func_img2img
from src.datasets.image2image.edit_datasets import ReconstructDataset


with read_base():
    from .processors import prompt_template, tokenizer, image_size, pad_index


dataset = dict(type=ReconstructDataset,
               image_size=image_size,
               cap_folder='data/megalith-10m/captions',
               data_path='data/megalith-10m/megalith10m_all.json',
               image_folder='data/megalith-10m/raw',
               prompt_template=prompt_template,
               tokenizer=tokenizer)

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    collate_fn=dict(type=collate_func_img2img,
                    pad_index=pad_index)
)
