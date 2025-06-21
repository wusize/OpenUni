
# Laion6m

This dataset is originally from [dclure/laion-aesthetics-12m-umap](https://huggingface.co/datasets/dclure/laion-aesthetics-12m-umap).
We re-captioned 6m of them whose width-height ratio is within 3.

## Download
Since we are not authorized to re-distribute the images of Laion dataset, here we only provide the url and captions
in [wusize/laion6m_recap](https://huggingface.co/datasets/wusize/laion6m_recap).
```shell
cd /path/to/OpenUni
huggingface-cli download wusize/laion6m_recap --local-dir data/laion6m/parquets --repo-type dataset
```
It is recommended to download the images using [img2dataset](https://github.com/rom1504/img2dataset). After downloading
the images, please arrange the data in the following format.

```text
OpenUni/
├── data
    ├── laion6m
        ├── raw
            ├── 00000000
                ├── 00000001.jpg
                ├── 00000001.json
                ├── 00000002.jpg
                ├── 00000002.json
            
            ├── 00000001
        |── parquets
        |── data.json
```

The file `data.json` would contain the paths to images (.jpg) and annotations (.json) with a list of dicts:

```
[{'image': '000000/0000001.jpg', 'annotation': '000000/0000001.json'},
{'image': '000000/0000002.jpg', 'annotation': '000000/0000002.json'},
]
```

## Set config

```python
from src.datasets.text2image.caption_datasets import CaptionDataset
from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from xtuner.dataset import ConcatDataset

with read_base():
    from .processors import prompt_template, tokenizer, image_size, pad_index

max_length = 128

dataset = dict(type=CaptionDataset,
               image_size=image_size,
               cap_source='re_caption',
               cap_folder='data/laion6m/raw',
               data_path='data/laion6m/data.json',
               image_folder='data/laion6m/raw',
               unconditional=0.1,
               prompt_template=prompt_template,
               ceph_folder=None,
               ceph_config=None,
               tokenizer=tokenizer,
               max_length=max_length)
```