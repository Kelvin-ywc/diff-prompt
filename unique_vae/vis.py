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
    # mask = torch.clamp(127.5 * mask + 128.0, 0, 255).detach().squeeze(dim=0).squeeze(0).to("cpu", dtype=torch.uint8).numpy()
    mask = torch.clamp(255.0 * mask, 0, 255).detach().squeeze(dim=0).squeeze(0).to("cpu", dtype=torch.uint8).numpy()
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
                    default='configs/mask_vae_v2_test.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = vae_models[config['model_params']['name']](**config['model_params'])

# vae_model = VAEXperiment.load_from_checkpoint(vae_model=model, params=config['exp_params'], checkpoint_path='logs/MaskVanillaVAE_lr_0005/version_24/checkpoints/epoch=119-step=79559.ckpt')
# vae_model = VAEXperiment.load_from_checkpoint(vae_model=model, params=config['exp_params'], checkpoint_path='logs/MaskVanillaVAE_e100/version_0/checkpoints/epoch=99-step=8299.ckpt')
# vae_model = VAEXperiment.load_from_checkpoint(vae_model=model, params=config['exp_params'], checkpoint_path='logs/MaskVanillaVAE_V2_best_norm/version_68/checkpoints/epoch=140-step=23405.ckpt')
vae_model = VAEXperiment.load_from_checkpoint(vae_model=model, params=config['exp_params'], checkpoint_path='logs/MaskVanillaVAE_V2/version_0/checkpoints/epoch=128-step=21413.ckpt')


annotation_files = "/home1/yanweicai/DATA/CV/coco/annotations/refcoco-unc/instances.json"
test_annotation_files = [annotation_files, annotation_files]
test_dataset_types = ['testA', 'testB']

dataset = SegmentMaskData(test_annotation_files, test_dataset_types)

dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=0, shuffle=False)
idx = 0
# for i, mask in enumerate(dataloader):
#     if idx == 10:
#         break
#     idx += 1
#     # org mask visualization
#     mask_org = mask_to_image(mask[0])
#     mask_org.save(f"vis/mask_org_{i}.png")
#     print(f"mask_org_{i}.png saved.")
    
#     img = mask[0]
#     mask_new = model.generate(img)
#     mask_new = mask_to_image(mask_new)
#     mask_new.save(f"vis/mask_new_{i}.png")
#     print(f"mask_new_{i}.png saved.")
# samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

vae_model.to(torch.device(0))
print(1)