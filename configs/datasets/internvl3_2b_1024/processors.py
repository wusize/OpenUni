from transformers import AutoImageProcessor, AutoTokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


internvl3_model_name_or_path = "OpenGVLab/InternVL3-2B"
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
tokenizer_kwargs = dict(add_special_tokens=True)

pad_index = 0
image_length = 256
image_size = 1024

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=internvl3_model_name_or_path,
    trust_remote_code=True,
    padding_side='right')


image_processor = dict(type=AutoImageProcessor.from_pretrained,
                       pretrained_model_name_or_path=internvl3_model_name_or_path)
