import torch
import numpy as np
from einops import rearrange
from src.datasets.utils import crop2square
from src.datasets.text2image.caption_datasets import CaptionDataset


class ImageEditDataset(CaptionDataset):
    def _process_image(self, image):
        image = image.resize(size=(self.image_size, self.image_size))
        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')
        return pixel_values

    def _process_text(self, text):
        prompt_template = self.prompt_template
        image_tokens = prompt_template['IMG_START_TOKEN'] + \
                       prompt_template['IMG_CONTEXT_TOKEN'] * self.image_length + \
                       prompt_template['IMG_END_TOKEN']
        prompt = f'{image_tokens}\n{text}'
        prompt = self.prompt_template['INSTRUCTION'].format(input=prompt)
        prompt += prompt_template['IMG_START_TOKEN']
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt', **self.tokenizer_kwargs)[0]

        return dict(input_ids=input_ids)

    def __getitem__(self, idx):
        if self.debug:
            idx = 0
        try:
            data_sample = self.data_list[idx]
            source_image = self._read_image(data_sample['source_image']).convert('RGB')
            target_image = self._read_image(data_sample['target_image']).convert('RGB')
            prompt = self._read_json(data_sample['annotation'])[self.cap_source]

            pixel_values_src = self._process_image(source_image)
            pixel_values = self._process_image(target_image)

            data = self._process_text(prompt)

            data.update(
                pixel_values_src=pixel_values_src, pixel_values=pixel_values,
                image_dir=self.local_folder, image_file=data_sample['image'],
                type='image2image')

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()


class ReconstructDataset(ImageEditDataset):
    def _process_image(self, image):
        image = crop2square(image)
        image = image.resize(size=(self.image_size, self.image_size))
        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')

        return pixel_values

    def __getitem__(self, idx):
        if self.debug:
            idx = 0
        try:
            data_sample = self.data_list[idx]
            image = self._read_image(data_sample['image']).convert('RGB')
            prompt = "Keep the image as it is."
            pixel_values = pixel_values_src = self._process_image(image)

            data = self._process_text(prompt)

            data.update(
                pixel_values_src=pixel_values_src, pixel_values=pixel_values,
                image_dir=self.local_folder, image_file=data_sample['image'],
                type='image2image')

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()
