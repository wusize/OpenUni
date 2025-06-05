
## Text-to-Image
**MMUni-B-512**

```shell
cd /path/to/OpenUni
export PYTHONPATH=.
python scripts/text2image.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint /path/to/your/ckpt \
             --prompt "a dog on the left and a cat on the right." \
             --output /path/to/save/the/result.jpg \
             --height 512 --width 512 \
             --seed 42
```

**MMUni-L-512**

```shell
cd /path/to/OpenUni
export PYTHONPATH=.
python scripts/text2image.py configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint /path/to/your/ckpt \
             --prompt "a dog on the left and a cat on the right." \
             --output /path/to/save/the/result.jpg \
             --height 512 --width 512 \
             --seed 42
```

**MMUni-L-1024**

```shell
cd /path/to/OpenUni
export PYTHONPATH=.
python scripts/text2image.py configs/models/openuni_l_internvl3_2b_sana_1_6b_1024_hf.py --checkpoint /path/to/your/ckpt \
             --prompt "a dog on the left and a cat on the right." \
             --output /path/to/save/the/result.jpg \
             --height 1024 --width 1024 \
             --seed 42
```

## Image-to-Image

**MMUni-B-512**

```shell
cd /path/to/OpenUni
export PYTHONPATH=.
python scripts/image_edit.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint /path/to/your/ckpt \
             --prompt "Keep the image as it is." \
             --output /path/to/save/the/result.jpg \
             --height 512 --width 512 \
             --seed 42
```
