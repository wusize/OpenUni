
# jackyhate/text-to-image-2M
## Download

Download the images and captions by
```shell
cd /path/to/OpenUni
huggingface-cli download jackyhate/text-to-image-2M --local-dir data/text-to-image-2M/raw --repo-type dataset
```

We have also gathered the paths of the images and captions, and saved in json files:
```shell
huggingface-cli download wusize/text-to-image-2M --local-dir data/text-to-image-2M/data --repo-type dataset
```



```text
OpenUni/
├── data
    ├── text-to-image-2M
        ├── raw
            ├── data_1024_10K
            ├── data_512_2M
        ├── data
            ├── data_1024_10K.json
            ├── data_512_2M.json
```


## Extract

Then to extract the data samples from .tar files. To extract the images of 1024

```shell
cd data/text-to-image-2M/raw/data_1024_10K
mkdir data_000000
tar -xf data_000000.tar -C data_000000
```

To extract the images of 512, we use multiple processes to deal with all the .tar files

```shell
cd data/text-to-image-2M/raw/data_512_2M
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
```