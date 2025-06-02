
# RedCaps5M

This dataset is originally from [RedCaps12M](https://redcaps.xyz/download).
We successfully downloaded 5m images and re-captioned them. Since the original images are too large, we downsampled the
images so that their shortest edge is limited to 1024.

## Download

Download the images and captions by
```shell
cd /path/to/OpenUni
huggingface-cli download wusize/redcaps5m_resized --local-dir data/redcaps5m --repo-type dataset
```

```text
OpenUni/
├── data
    ├── redcaps5m
        ├── raw
        |── redcaps5m_data.json
```


## Extract Images and Captions

Then to extract the images from .tar files. To extract the images of 1024

```shell
cd data/redcaps5m/raw
vim extract.py
```
Write the following into extract.py

```python
import multiprocessing as mp
import argparse
import os
from tqdm import tqdm
from glob import glob
import subprocess


def single_process(tar_list,):
    for tar_file in tqdm(tar_list):
        subprocess.run(["tar", "-xf", tar_file, "--no-same-owner"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=-1, type=int)
    parser.add_argument('--num-processes', default=8, type=int)
    args = parser.parse_args()

    tar_files = sorted(glob(f'*.tar'))
    
    if args.end == -1:
        args.end = len(tar_files)

    tar_files = tar_files[args.start:args.end]

    num_tars = len(tar_files)
    num_processes = args.num_processes
    num_tars_per_process = num_tars // num_processes
    res = num_tars % num_processes
    if res > 0:
        num_processes += 1

    processes = [mp.Process(target=single_process,
                            args=(tar_files[process_id * num_tars_per_process:
                                            (process_id + 1) * num_tars_per_process]
                                  if process_id < num_processes - 1
                                  else tar_files[process_id * num_tars_per_process:],
                                  ))
                 for process_id in range(num_processes)]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

```

Then run the python file to extract all the tar files:

```shell
python extract.py --num-processes 8
# python extract.py --num-processes 8 --start 0 --end 32   # you can also process a part of the .tars in a single task and launch many tasks   
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
               cap_folder='data/redcaps5m/raw',
               data_path='data/redcaps5m/redcaps5m_data.json',
               image_folder='data/redcaps5m/raw',
               unconditional=0.1,
               prompt_template=prompt_template,
               ceph_folder=None,
               ceph_config=None,
               tokenizer=tokenizer,
               max_length=max_length)
```