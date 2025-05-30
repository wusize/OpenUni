# OpenUni: A Simple Baseline for Unified Multimodal Understanding and Generation

![](data/teaser.png)

> **[OpenUni: A Simple Baseline for Unified Multimodal Understanding and Generation](https://arxiv.org/abs/2505.23661)**
>
> Size Wu, Zhonghua Wu, Zerui Gong, Qingyi Tao, Sheng Jin, Qinyue Li, Wei Li, Chen Change Loy
>
> [![arXiv](https://img.shields.io/badge/arXiv-2505.23661-b31b1b.svg)](https://arxiv.org/abs/2505.23661)
> [![Bibtex](https://img.shields.io/badge/Cite-BibTeX-blue)](https://github.com/wusize/OpenUni?tab=readme-ov-file#-citation)

## Introduction

This is a repo under construction, named OpenUni, an open-source version of [MetaQuery](https://xichenpan.com/metaquery) for unifying multimodal understanding and generation.
Currently, we provide three model variants: OpenUni-B-512, OpenUni-L-512, OpenUni-L-1024. We will be updating this repo so stay tuned!

## Environment (just for reference)
```
mmengine
xtuner
transformers==4.45.2
torch==2.3.1
flash_attn==2.3.4
```



## 📚 Citation

If you find OpenUni useful for your research or applications, please cite our paper using the following BibTeX:

```bibtex
@article{wu2025openuni,
      title={OpenUni: A Simple Baseline for Unified Multimodal Understanding and Generation}, 
      author={Size Wu and Zhonghua Wu and Zerui Gong and Qingyi Tao and Sheng Jin and Qinyue Li and Wei Li and Chen Change Loy},
      year={2025},
      eprint={2505.23661},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.23661}, 
}
```

## 📜 License
This project is licensed under [NTU S-Lab License 1.0](LICENSE).


## 🙏 Acknowledgement
The project builds upon the following pioneering works:
- [SANA](https://github.com/NVlabs/Sana): We use SANA as our diffusion module, considering its efficiency and strong performance.
- [InternVL3](https://github.com/OpenGVLab/InternVL): We use the latest InternVL3 as our base multimodal LLM.
- [MetaQuery](https://xichenpan.com/metaquery): OpenUni is inspired by MetaQuery and is an open-source implementation of this work.
- [BLIP3-o](https://github.com/JiuhaiChen/BLIP3o): We thank the BLIP3-o team for releasing the precious high-quality tuning dataset.
