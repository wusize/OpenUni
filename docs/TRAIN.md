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
We have provided the code but do not ensure this works.
### OpenUni-B-512

```shell
cd /path/to/OpenUni
export PYTHONPATH=./:$PYTHONPATH
GPUS_PER_NODE=8 NNODES=1 bash scripts/train_ddp.sh \
     configs/pretrain/openuni_b_internvl3_1b_sana_0_6b_512_hf_image2image23m.py \
     --deepspeed deepspeed_zero2
```