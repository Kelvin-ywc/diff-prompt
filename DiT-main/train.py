# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
from sched import scheduler
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import json

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

from utils.data.datasets.datasets import Coco
from utils.data.batch_collate import BatchCollator
from utils.data.datasets.datasets import ImageObj

import clip
from unique_vae.experiment import VAEXperiment
from unique_vae.models import MaskVanillaVAE_V2
#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param

def toImageList(imgs :ImageObj) -> torch.tensor:
    return torch.stack([img.image for img in imgs], dim=0)

def merge_cfg(args, param):
    for key in param:
        setattr(args, key, param[key])
    return args
#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    param = load_json(args.config_file)
    args = merge_cfg(args, param)
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8 # 32
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    clip_model, preprocess = clip.load("ViT-B/32", device=device) # Create a CLIP model for extarcting image and text features


    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])

    # diffusion = create_diffusion(timestep_respacing="", diffusion_steps=50, noise_schedule='squaredcos_cap_v2')  # default: 1000 steps, linear noise schedule
    diffusion = create_diffusion(timestep_respacing="", diffusion_steps=100)
    # load ave model
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    import yaml
    with open('configs/vae/mask_vae_v2.yaml', 'r') as file:
        try:
            vae_config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    vae = MaskVanillaVAE_V2()
    # checkpoint_path = '/home1/yanweicai/workspace/prompt/PromptDiffusion/unique_vae/logs/MaskVanillaVAE_V2/version_0/checkpoints/epoch=128-step=21413.ckpt'
    # checkpoint_path = '/home1/yanweicai/workspace/prompt/PromptDiffusion/unique_vae/logs/MaskVanillaVAE_V2/version_20/checkpoints/epoch=157-step=13113.ckpt'
    # vae_experiment = VAEXperiment.load_from_checkpoint(vae_model=vae, params=vae_config['exp_params'], checkpoint_path=checkpoint_path)

    # vae = DDP(vae.to(device), device_ids=[rank])
    vae = AutoencoderKL.from_pretrained('oaaoaa/mask_vae').to(device)
    vae = vae.to(device)
    vae.eval()

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=0)
    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    # dataset = ImageFolder(args.data_path, transform=transform)
    dataset = Coco(cfg=args)

    collator = BatchCollator()

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({param['data_path']})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        # for x, y in loader:
        #     # forward
        #     x = x.to(device)
        #     y = y.to(device)
        #     with torch.no_grad():
        #         # Map input images to latent space + normalize latents:
        #         x = vae.encode(x).latent_dist.sample().mul_(0.18215)
        #     t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        #     model_kwargs = dict(y=y)
        #     loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
        for image, mask, caption in loader:
            image_tensor = toImageList(image)
            mask_tensor = toImageList(mask)

            image_tensor = image_tensor.to(device)
            mask_tensor = mask_tensor.to(device)
            with torch.no_grad():
                # mask_tensor = vae.encode(mask_tensor).latent_dist.sample().mul_(0.18215)
                encoding =vae.encode(mask_tensor).latent_dist
                mu, log_var = encoding.mean, encoding.logvar
                # mask_tensor = vae.reparameterize(mu, log_var).mul_(0.18215) # [32, 4096]
                # mask_tensor = vae.reparameterize(mu, log_var) # [32, 4096]
                mask_tensor = encoding.sample()

                # image_tensor = preprocess(image_tensor)
                image_features = clip_model.encode_image(image_tensor)
                caption = clip.tokenize(caption).to(device)
                text_features = clip_model.encode_text(caption)

            t = torch.randint(0, diffusion.num_timesteps, (image_tensor.shape[0],), device=device)
            model_kwargs = dict(image_features=image_features, text_features=text_features)
            loss_dict = diffusion.training_losses(model, mask_tensor, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            # if train_steps % args.ckpt_every == 0 and train_steps > 0:
            #     if rank == 0:
            #         checkpoint = {
            #             "model": model.module.state_dict(),
            #             "ema": ema.state_dict(),
            #             "opt": opt.state_dict(),
            #             "args": args
            #         }
            #         checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
            #         torch.save(checkpoint, checkpoint_path)
            #         logger.info(f"Saved checkpoint to {checkpoint_path}")
            #     dist.barrier()
        if (epoch+1) % 5==0:
            if rank == 0:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            dist.barrier()
    model.eval()  # important! This disables randomized embedding dropout
    scheduler.step()
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--config-file", type=str, default="configs/refcoco/train_refcoco.json")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-My/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=224)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global-batch-size", type=int, default=2)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--vae-path", type=str, default='home1/yanweicai/workspace/prompt/PromptDiffusion/PyTorch-VAE/logs/MaskVanillaVAE/version_0/checkpoints/epoch=98-step=32867.ckpt')
    parser.add_argument("--num-workers", type=int, default=48)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    args = parser.parse_args()
    main(args)
