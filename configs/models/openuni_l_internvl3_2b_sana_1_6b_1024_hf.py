import torch
from src.models.openuni.internvl3_sana_hf import OpenUniInternVL3SANAHF
from src.models.internvl3.modeling_internvl_chat import InternVLChatModel
from diffusers import (AutoencoderDC, SanaTransformer2DModel,
                       DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler)

from mmengine.config import read_base

with read_base():
    from ..datasets.internvl3_2b_1024.processors import \
        prompt_template, tokenizer, internvl3_model_name_or_path, image_size


sana_model_name_or_path = f"Efficient-Large-Model/SANA1.5_1.6B_{image_size}px_diffusers"

model = dict(type=OpenUniInternVL3SANAHF,
             num_queries=256,
             connector=dict(
                 hidden_size=1536,
                 intermediate_size=8960,
                 num_hidden_layers=6,
                 _attn_implementation='flash_attention_2',
                 num_attention_heads=24,),
             lmm=dict(
                 type=InternVLChatModel.from_pretrained,
                 pretrained_model_name_or_path=internvl3_model_name_or_path,
                 torch_dtype=torch.bfloat16,
                 low_cpu_mem_usage=True,
                 use_flash_attn=True,
             ),
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
