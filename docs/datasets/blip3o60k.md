
# BLIP3o/BLIP3o-60k
## Download

Download the images and captions by
```shell
cd /path/to/OpenUni
huggingface-cli download BLIP3o/BLIP3o-60k --local-dir data/BLIP3o-60k/raw --repo-type dataset
```

We have also gathered the paths of the images and captions, and saved in json files:
```shell
huggingface-cli download wusize/BLIP3o-60k --local-dir data/BLIP3o-60k --repo-type dataset --include "*.json"
```


```text
OpenUni/
├── data
    ├── BLIP3o-60k
        ├── raw
        ├── data.json
```


## Extract
```shell
cd data/BLIP3o-60k/raw
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
        folder = tar_file[:-4]
        os.makedirs(folder, exist_ok=True)
        subprocess.run(["tar", "-xf", tar_file, "-C", folder, "--no-same-owner"])


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
python extract.py --num-processes 11
```

## Set config

```python
from src.datasets.text2image.blip3_o import BLIP3oDataset
from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from xtuner.dataset import ConcatDataset

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

```
