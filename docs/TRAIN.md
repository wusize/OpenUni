
## Extract latents
To increase training efficiency, it is necessary to extract and save the VAE latents.


### 512x512 Images

```shell
torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/process_data/tokenize_images_ddp_dc32_hf.py \
    $CONFIG \
    --batch_size 32
```

Set `CONFIG` to the following dataset configs:
```shell
configs/datasets/internvl3_2b_512/laion6m.py
configs/datasets/internvl3_2b_512/megalith10m.py
configs/datasets/internvl3_2b_512/redcaps5m.py
configs/datasets/internvl3_2b_512/text2image2m.py
configs/datasets/internvl3_2b_512/blip3o60k.py
```


### 1024X1024 Images

```shell
torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/process_data/tokenize_images_ddp_dc32_hf.py \
    $CONFIG \
    --batch_size 8
```

Set `CONFIG` to the following dataset configs:
```shell
configs/datasets/internvl3_2b_1024/laion6m.py
configs/datasets/internvl3_2b_1024/megalith10m.py
configs/datasets/internvl3_2b_1024/redcaps5m.py
configs/datasets/internvl3_2b_1024/text2image2m.py
configs/datasets/internvl3_2b_1024/blip3o60k.py
```



## Text-to-Image
### OpenUni-B-512

**Pretrain**

Multiple nodes are used for pretraining, so make sure you node rank, master address and master port has been written
into env variables `NODE_RANK`, `MASTER_ADDR` and `MASTER_PORT`.
```shell
cd /path/to/OpenUni
export PYTHONPATH=./:$PYTHONPATH
GPUS_PER_NODE=8 NNODES=2 bash scripts/train_ddp.sh \
     configs/pretrain/openuni_b_internvl3_1b_sana_0_6b_512_hf_text2image23m.py \
     --deepspeed deepspeed_zero2
```

**Finetune**

Modify the finetune config to set the path of your pretrained weights:
```python
model.pretrained_pth = 'path/to/the/pretrained/checkpoint.pth' 
# e.g., work_dirs/openuni_b_internvl3_1b_sana_0_6b_512_hf_text2image23m/iter_100000.pth
```

```shell
cd /path/to/OpenUni
export PYTHONPATH=./:$PYTHONPATH
GPUS_PER_NODE=8 NNODES=1 bash scripts/train_ddp.sh \
     configs/finetune/openuni_l_internvl3_1b_sana_0_6b_512_hf_blip3o60k.py \
     --deepspeed deepspeed_zero2
```



### OpenUni-L-512

**Pretrain**

Multiple nodes are used for pretraining, so make sure you node rank, master address and master port has been written
into env variables `NODE_RANK`, `MASTER_ADDR` and `MASTER_PORT`.
```shell
cd /path/to/OpenUni
export PYTHONPATH=./:$PYTHONPATH
GPUS_PER_NODE=8 NNODES=2 bash scripts/train_ddp.sh \
     configs/pretrain/openuni_l_internvl3_2b_sana_1_6b_512_hf_text2image23m.py \
     --deepspeed deepspeed_zero2
```

**Finetune**

Modify the finetune config to set the path of your pretrained weights:
```python
model.pretrained_pth = 'path/to/the/pretrained/checkpoint.pth' 
# e.g., work_dirs/openuni_l_internvl3_2b_sana_1_6b_512_hf_text2image23m/iter_100000.pth
```

```shell
cd /path/to/OpenUni
export PYTHONPATH=./:$PYTHONPATH
GPUS_PER_NODE=8 NNODES=1 bash scripts/train_ddp.sh \
     configs/finetune/openuni_l_internvl3_2b_sana_1_6b_512_hf_blip3o60k.py \
     --deepspeed deepspeed_zero2
```



### OpenUni-L-1024

**Pretrain**

Multiple nodes are used for pretraining, so make sure you node rank, master address and master port has been written
into env variables `NODE_RANK`, `MASTER_ADDR` and `MASTER_PORT`.
```shell
cd /path/to/OpenUni
export PYTHONPATH=./:$PYTHONPATH
GPUS_PER_NODE=8 NNODES=4 bash scripts/train_ddp.sh \
     configs/pretrain/openuni_l_internvl3_2b_sana_1_6b_1024_hf_text2image23m.py \
     --deepspeed deepspeed_zero2
```

**Finetune**

Modify the finetune config to set the path of your pretrained weights:
```python
model.pretrained_pth = 'path/to/the/pretrained/checkpoint.pth' 
# e.g., configs/pretrain/openuni_l_internvl3_2b_sana_1_6b_1024_hf_text2image23m.pth
```

Multiple nodes are used for pretraining, so make sure you node rank, master address and master port has been written
into env variables `NODE_RANK`, `MASTER_ADDR` and `MASTER_PORT`.
```shell
cd /path/to/OpenUni
export PYTHONPATH=./:$PYTHONPATH
GPUS_PER_NODE=8 NNODES=2 bash scripts/train_ddp.sh \
     configs/finetune/openuni_l_internvl3_2b_sana_1_6b_1024_hf_blip3o60k.py \
     --deepspeed deepspeed_zero2
```


## Image-to-Image (Under Development)
We have provided the code but do not guarantee this works.
### OpenUni-B-512

```shell
cd /path/to/OpenUni
export PYTHONPATH=./:$PYTHONPATH
GPUS_PER_NODE=8 NNODES=1 bash scripts/train_ddp.sh \
     configs/pretrain/openuni_b_internvl3_1b_sana_0_6b_512_hf_image2image10m.py \
     --deepspeed deepspeed_zero2
```