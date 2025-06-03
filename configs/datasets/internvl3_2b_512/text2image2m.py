from src.datasets.text2image.caption_datasets import CaptionDataset
from src.datasets.collate_functions import collate_func_gen
from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from xtuner.dataset import ConcatDataset

with read_base():
    from .processors import prompt_template, tokenizer, image_size, pad_index

max_length = 128

t2i_2m = dict(type=CaptionDataset,
              image_size=image_size,
              cap_source='prompt',
              data_path='data/text-to-image-2M/data/data_512_2M.json',
              cap_folder='data/text-to-image-2M/raw/data_512_2M',
              image_folder='data/text-to-image-2M/raw/data_512_2M',
              unconditional=0.1,
              prompt_template=prompt_template,
              ceph_folder=None,
              ceph_config=None,
              tokenizer=tokenizer,
              max_length=max_length)

t2i_10k = dict(type=CaptionDataset,
               image_size=image_size,
               cap_source='prompt',
               data_path='data/text-to-image-2M/data/data_1024_10K.json',
               cap_folder='data/text-to-image-2M/raw/data_1024_10K',
               image_folder='data/text-to-image-2M/raw/data_1024_10K',
               unconditional=0.1,
               prompt_template=prompt_template,
               ceph_folder=None,
               ceph_config=None,
               tokenizer=tokenizer,
               max_length=max_length)

dataset = dict(
    type=ConcatDataset,
    datasets=[t2i_2m, t2i_10k]
)

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    collate_fn=dict(type=collate_func_gen,
                    pad_index=pad_index)
)
