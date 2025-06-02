import torch
import os
from src.datasets.text2image.caption_datasets import CaptionDataset


class BLIP3oDataset(CaptionDataset):
    def __getitem__(self, idx):
        if self.debug:
            idx = 0
        try:
            data_sample = self.data_list[idx]

            if self.image_tokens_folder is not None:
                image_tokens = torch.load(os.path.join(self.image_tokens_folder,
                                                       data_sample['image'] + '.pt')).long()
                data = dict(image_tokens=image_tokens)
            elif self.latents_ceph_folder is not None:
                image_latents = torch.load(
                    self._read_ceph(
                        os.path.join(
                            self.latents_ceph_folder, data_sample['image'] + '.pt'
                        )
                    )
                )
                data = dict(image_latents=image_latents)
            elif self.image_latents_folder is not None:
                image_latents = torch.load(os.path.join(self.image_latents_folder,
                                                        data_sample['image'] + '.pt'))
                data = dict(image_latents=image_latents)
            else:
                image = self._read_image(data_sample['image']).convert('RGB')
                data = self._process_image(image)

            with open(os.path.join(self.cap_folder, data_sample['annotation']), 'r') as f:
                caption = f.read().strip()

            # print(caption)

            data.update(self._process_text(caption))
            data.update(image_dir=self.image_folder, image_file=data_sample['image'],
                        type='text2image')

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()
