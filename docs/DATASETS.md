## Pretrain

Some of our pretraining data are directly from open-source datasets, 
including [megalith10m.md](datasets/megalith10m.md) and [text2image2m.md](datasets/text2image2m.md). Additionally, we
collected and re-captioned 6m [Laion](https://huggingface.co/datasets/dclure/laion-aesthetics-12m-umap) images and 
5m [RedCaps](https://redcaps.xyz) images. Due to legal concerns, we are unable to redistribute these images.
Please refer to [laion6m.md](datasets/laion6m.md) and [redcaps5m.md](datasets/redcaps5m.md) for
the urls and captions.

Considering that RedCaps images are super large and hard to obtain, we give an alternative dataset for this
part of data, i.e., CC12M. Again, we do not have the right to redistribute the images but only upload our captions.
So we recommend to download the images from other sources such as [pixparse/cc12m-wds](https://huggingface.co/datasets/pixparse/cc12m-wds).
There is a guidance to prepare the CC12M dataset in [cc12m.md](datasets/cc12m.md)

## Finetune

Currently, we use [BLIP3o/BLIP3o-60k](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k) as the finetuning dataset
to compare with prior models such as BLIP3o-4B. There is a guidance to prepare this dataset in [blip3o60k.md](datasets/blip3o60k.md).
However, [BLIP3o/BLIP3o-60k](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k) contains 10K data samples that use
the exactly same prompt templates as GenEval, thus drastically boosting performance on this benchmark. If you have
concerns over data leakage, it is recommended to use [text2image2m.md](datasets/text2image2m.md) for finetuning.