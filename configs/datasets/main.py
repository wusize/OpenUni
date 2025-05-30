import argparse
from tqdm import tqdm
from src.runners.custom_runner import CustomRunner
from mmengine.config import Config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()
    config = Config.fromfile(args.config)
    dataloader = CustomRunner.build_dataloader(config.train_dataloader)
    print(len(dataloader.dataset))
    for data in tqdm(dataloader):
        print(data['data'].keys(), flush=True)
