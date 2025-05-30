## Env (just for reference)
```
mmengine
xtuner
transformers==4.45.2
torch==2.3.1
flash_attn==2.3.4
```


## Inference


```shell
export PYTHONPATH=.
python scripts/text2image.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint /path/to/your/ckpt \
             --prompt "a dog on the left and a cat on the right." \
             --output /path/to/save/the/result.jpg \
             --height 512 -width 512 \
             --seed 42
```


## Train

```shell
export PYTHONPATH=.
torchrun --nproc_per_node=1 --nnodes=8 scripts/train.py \ 
           configs/finetune/openuni_l_internvl3_1b_sana_0_6b_512_hf_blip3o60k.py \ 
           --launcher pytorch  --deepspeed deepspeed_zero2
```


## Evaluation

### GenEval
```shell
export PYTHONPATH=.
accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py \ 
         --checkpoint /path/to/your/ckpt  --batch_size 8  \ 
         --output /path/to/save/results \
         --height 512 -width 512 \
         --seed 42
```

### DPG Bench
```shell
export PYTHONPATH=.
accelerate launch scripts/evaluation/dpg_bench.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py \ 
         --checkpoint /path/to/your/ckpt  --batch_size 8  \ 
         --output /path/to/save/results \
         --height 512 -width 512 \
         --seed 42
```
