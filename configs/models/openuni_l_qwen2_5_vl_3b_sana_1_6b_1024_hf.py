import torch
from src.models.openuni.qwen2_5_vl_sana_hf import OpenUniQwen25VLSANAHF
from transformers import Qwen2_5_VLForConditionalGeneration
from diffusers import (AutoencoderDC, SanaTransformer2DModel,
                       DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler)

from mmengine.config import read_base

with read_base():
    from ..datasets.qwen2_5_vl_3b_1024.processors import \
        prompt_template, tokenizer, qwen2_5_vl_model_name_or_path, image_size


sana_model_name_or_path = f"Efficient-Large-Model/SANA1.5_1.6B_{image_size}px_diffusers"

model = dict(type=OpenUniQwen25VLSANAHF,
             num_queries=256,
             connector=dict(
                 hidden_size=1536,
                 intermediate_size=8960,
                 num_hidden_layers=6,
                 _attn_implementation='flash_attention_2',
                 num_attention_heads=24,),
             lmm=dict(type=Qwen2_5_VLForConditionalGeneration.from_pretrained,
                      pretrained_model_name_or_path=qwen2_5_vl_model_name_or_path,
                      torch_dtype=torch.bfloat16,
                      attn_implementation="flash_attention_2", ),
             vae=dict(type=AutoencoderDC.from_pretrained,
                      pretrained_model_name_or_path='mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers',
                      torch_dtype=torch.bfloat16),
             transformer=dict(type=SanaTransformer2DModel.from_pretrained,
                              pretrained_model_name_or_path=sana_model_name_or_path,
                              subfolder="transformer",
                              torch_dtype=torch.bfloat16),
             train_scheduler=dict(type=FlowMatchEulerDiscreteScheduler.from_pretrained,
                                  pretrained_model_name_or_path=sana_model_name_or_path,
                                  subfolder="scheduler"),
             test_scheduler=dict(type=DPMSolverMultistepScheduler.from_pretrained,
                                 pretrained_model_name_or_path=sana_model_name_or_path,
                                 subfolder="scheduler"),
             tokenizer=tokenizer,
             prompt_template=prompt_template,
             lora_modules=None,
             freeze_lmm=True,
             freeze_transformer=True
             )
