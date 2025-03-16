import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from .models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset, Cfg, SegmentMaskData, SegmentMaskModule
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader
from PIL import Image

from torchvision import transforms

# single image
def mask_to_image(mask: torch.Tensor) -> Image:
    mask = torch.clamp(127.5 * mask + 128.0, 0, 255).squeeze(dim=0).squeeze(0).to("cpu", dtype=torch.uint8).numpy()
    # mask = mask.squeeze(dim=0)
    # mask *= 255
    # toPIL = transforms.ToPILImage() 
    # mask = toPIL(mask)
    mask = Image.fromarray(mask)
    return mask

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/mask_vae_test.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

# model = vae_models[config['model_params']['name']](**config['model_params'])

# vae_model = VAEXperiment.load_from_checkpoint(vae_model=model, params=config['exp_params'], checkpoint_path='PyTorch-VAE/logs/MaskVanillaVAE/version_0/checkpoints/epoch=98-step=32867.ckpt')

annotation_files = "/home1/yanweicai/DATA/CV/coco/annotations/refcoco-unc/instances.json"
# test_annotation_files = [annotation_files, annotation_files]
# test_dataset_types = ['testA', 'testB']
train_annotation_files = [annotation_files]
train_dataset_types = ['train']

dataset = SegmentMaskData(train_annotation_files, train_dataset_types)

dataloader = DataLoader(dataset=dataset, batch_size=128, num_workers=0, shuffle=False)
mean = torch.zeros(1)
std = torch.zeros(1)
batch_count = 0
for i, mask in enumerate(dataloader):
    batch_count += 1
    mask = mask[0]
    mean[0] += mask[:,0,:,:].mean()
    std[0] += mask[:,0, :, :].std()

mean.div_(batch_count)
std.div_(batch_count)
print(f'mean: {mean.numpy()}, std: {std.numpy()}, num of images: {len(dataset)}')
# mean: [0.11714055], std: [0.29401192]
# mean: [0.00089937], std: [0.00247348]
# mean: [0.11487043], std: [0.31592062], num of images: 42404