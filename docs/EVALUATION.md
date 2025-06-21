## Evaluation
For now we only provide inference code, please turn to the official repos of the benchmarks to calculate final performance.

### GenEval
```shell
export PYTHONPATH=.
accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py \ 
         --checkpoint /path/to/your/ckpt  --batch_size 8  \ 
         --output /path/to/save/results \
         --height 512 --width 512 \
         --seed 42
```

### DPG Bench
```shell
export PYTHONPATH=.
accelerate launch scripts/evaluation/dpg_bench.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py \ 
         --checkpoint /path/to/your/ckpt  --batch_size 8  \ 
         --output /path/to/save/results \
         --height 512 --width 512 \
         --seed 42
```



### WISE Bench
```shell
export PYTHONPATH=.
accelerate launch scripts/evaluation/wise.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py \ 
         --checkpoint /path/to/your/ckpt  --batch_size 8  \ 
         --output /path/to/save/results \
         --height 512 --width 512 \
         --seed 42
```
