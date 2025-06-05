
# CC12M
The images of CC12M is from [pixparse/cc12m-wds](https://huggingface.co/datasets/pixparse/cc12m-wds). 
This captions are from [tomg-group-umd/pixelprose](https://huggingface.co/datasets/tomg-group-umd/pixelprose), which
are re-captioned to more concise and generation-oriented prompts.



## Download

Download the images by
```shell
cd /path/to/OpenUni
huggingface-cli download pixparse/cc12m-wds --local-dir data/cc12m/raw --repo-type dataset
```


Download the captions by
```shell
cd /path/to/OpenUni
huggingface-cli download wusize/cc12m_recap --local-dir data/cc12m --repo-type dataset
```

```text
OpenUni/
├── data
    ├── cc12m
        ├── raw
        |── captions
        |── data.json
```


## Extract Images

Then to extract the images from .tar files. To extract the images of 1024

```shell
cd data/cc12m/raw
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
        folder = folder.split('-')[-1]
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
python extract.py --num-processes 8
# python extract.py --num-processes 8 --start 0 --end 32   # you can also process a part of the .tars in a single task and launch many tasks   
```


## Extract Captions

```shell
cd data/cc12m/captions
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
               cap_folder='data/cc12m/captions',
               data_path='data/cc12m/data.json',
               image_folder='data/cc12m/raw',
               unconditional=0.1,
               prompt_template=prompt_template,
               ceph_folder=None,
               ceph_config=None,
               tokenizer=tokenizer,
               max_length=max_length)
```