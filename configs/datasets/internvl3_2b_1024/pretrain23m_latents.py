from src.datasets.collate_functions import collate_func_gen_latents
from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from xtuner.dataset import ConcatDataset


with read_base():
    from .processors import image_size, pad_index
    from .redcaps5m_latents import dataset as redcaps5m_latents
    from .laion6m_latents import dataset as laion6m_dataset
    from .text2image2m_latents import dataset as text2image2m_dataset
    from .megalith10m_latents import dataset as megalith10m_dataset


dataset = dict(
    type=ConcatDataset,
    datasets=[redcaps5m_latents, laion6m_dataset, text2image2m_dataset, megalith10m_dataset],
)


train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    collate_fn=dict(type=collate_func_gen_latents,
                    pad_index=pad_index)
)
