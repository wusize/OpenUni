import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from PIL import Image
import numpy as np
from xtuner.dataset.utils import expand2square
from einops import rearrange


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

from src.models.internvl3.modeling_internvl_chat import InternVLChatModel

pretrained_model_name_or_path = "OpenGVLab/InternVL3-8B"

model = InternVLChatModel.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,).cuda()


tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side='right')


image = Image.open('data/view.jpg')
image = expand2square(image, (127, 127, 127))
image = torch.from_numpy(np.array(image)).float() / 255
image_mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 3)
image_std = torch.tensor(IMAGENET_STD).view(1, 1, 3)
image = (image - image_mean) / image_std

image = rearrange(image, 'h w c -> c h w')[None].bfloat16().cuda()

image = F.interpolate(image, size=(448, 448), mode='bilinear')

generation_config = dict(max_new_tokens=128, do_sample=True, pad_token_id=0)

question = '<image>\nPlease describe the image in a short, vivid, and visually rich sentence.'

response = model.chat(tokenizer, image, question, generation_config)

print(response)


prompt_template = dict(
    IMG_START_TOKEN='<img>',
    IMG_END_TOKEN='</img>',
    IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
    SYSTEM='<|im_start|>system\n{system}<|im_end|>\n',
    INSTRUCTION=('<|im_start|>user\n{input}<|im_end|>\n'
                 '<|im_start|>assistant\n'),
    SUFFIX='<|im_end|>',
    SUFFIX_AS_EOS=True,
    SEP='\n',
    STOP_WORDS=['<|im_end|>'],
    GENERATION='Generate an image: {input}',
    CFG='Generate an image.'
)


question = '<image>\nCan you remove the wooden dock in the image? Produce the updated image using your image generator.'
SYSTEM = 'You are now equipped with an image generator so you can produce images, reply <img><|im_end|> whenever you are going to use the image generator.'

input_txt = prompt_template['SYSTEM'].format(system=SYSTEM) + prompt_template['INSTRUCTION'].format(input=question)
# input_txt = prompt_template['INSTRUCTION'].format(input=question)
img_context_token_id = tokenizer.convert_tokens_to_ids(prompt_template['IMG_CONTEXT_TOKEN'])
img_start_token_id = tokenizer.convert_tokens_to_ids(prompt_template['IMG_START_TOKEN'])
img_end_token_id = tokenizer.convert_tokens_to_ids(prompt_template['IMG_END_TOKEN'])

image_tokens = prompt_template['IMG_START_TOKEN'] + prompt_template['IMG_CONTEXT_TOKEN'] * 256 + prompt_template['IMG_END_TOKEN']
input_txt = input_txt.replace('<image>', image_tokens, 1)

model_inputs = tokenizer(input_txt, return_tensors='pt')
input_ids = model_inputs['input_ids'].cuda()
attention_mask = model_inputs['attention_mask'].cuda()

input_embeds = model.language_model.get_input_embeddings()(input_ids)

selected = input_ids == img_context_token_id
vit_embeds = model.extract_feature(image)
input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.flatten(0, 1)

generation_config['eos_token_id'] = tokenizer.convert_tokens_to_ids('<|im_end|>')


outputs = model.language_model.generate(
    inputs_embeds=input_embeds,
    attention_mask=attention_mask,
    **generation_config,
    use_cache=True)

response = tokenizer.decode(outputs[0], skip_special_tokens=False)


print(response)
