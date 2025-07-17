# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from utils.data.datasets.datasets import Coco
from utils.data.batch_collate import BatchCollator
from utils.data.datasets.datasets import ImageObj

import clip
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import json

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param

def merge_cfg(args, param):
    for key in param:
        setattr(args, key, param[key])
    return args

def toImageList(imgs :ImageObj) -> torch.tensor:
    return torch.stack([img.image for img in imgs], dim=0)

def main(args):
    param = load_json(args.config_file)
    args = merge_cfg(args, param)
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    # diffusion = create_diffusion(str(args.num_sampling_steps))
    diffusion = create_diffusion(timestep_respacing="ddim25", diffusion_steps=100, noise_schedule='squaredcos_cap_v2')
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae = AutoencoderKL.from_pretrained("oaaoaa/mask_vae").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

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
        batch_size=32,
        shuffle=False,
        sampler=sampler,
        num_workers=48,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({param['data_path']})")

    clip_model, preprocess = clip.load("ViT-B/32", device=device) 

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for image, mask, caption in loader:
        c = caption
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg

            model_kwargs = dict(y=y)
        else:
            image_tensor = toImageList(image)
            mask_tensor = toImageList(mask)

            image_tensor = image_tensor.to(device)
            mask_tensor = mask_tensor.to(device)

            with torch.no_grad():
                # image_tensor = preprocess(image_tensor)
                image_features = clip_model.encode_image(image_tensor)
                caption = clip.tokenize(caption).to(device)
                text_features = clip_model.encode_text(caption)
            model_kwargs = dict(image_features=image_features, text_features=text_features)

            
            sample_fn = model.forward
        import time
        # Sample images:
        time_start = time.time()  # 记录开始时间
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        time_end = time.time()  # 记录结束时间
        time_sum = time_end - time_start 
        print(f'ddpm time: {time_sum}')

        samples_ddim, samples_ddim_all = diffusion.ddim_sample_loop(sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device)
        time_end_ddim = time.time()
        time_sum = time_end_ddim - time_end
        print(f'ddim time: {time_sum}')
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples_ddim_all_decoded = []
        for samples_ddim in samples_ddim_all:
            sample_ddim_decoded = vae.decode(samples_ddim).sample
            sample_ddim_decoded = torch.clamp(255.0 * sample_ddim_decoded, 0, 255).squeeze(1).to("cpu", dtype=torch.uint8).numpy()
            samples_ddim_all_decoded.append(sample_ddim_decoded)
        # samples = vae.decode(samples / 0.18215).sample
        samples = vae.decode(samples).sample
        # samples_ddim = vae.decode(samples_ddim).sample
        # samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        samples = torch.clamp(255.0 * samples, 0, 255).squeeze(1).to("cpu", dtype=torch.uint8).numpy()
        # samples_ddim = torch.clamp(255.0 * samples_ddim, 0, 255).squeeze(1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, (sample) in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            local_index = i * dist.get_world_size() + rank
            mask = (mask_tensor[local_index]*255).squeeze(0).to("cpu", dtype=torch.uint8).numpy()
            Image.fromarray(mask).save(f"{sample_folder_dir}/{index:06d}_mask_{c[local_index]}.png")
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}_{c[local_index]}.png")
            # Image.fromarray(sample_ddim).save(f"{sample_folder_dir}/{index:06d}_ddim_{c[local_index]}.png")
            for step_idx, sample_ddim_decoded in enumerate(samples_ddim_all_decoded):
                Image.fromarray(sample_ddim_decoded[i]).save(f"{sample_folder_dir}/{index:06d}_ddim_{c[local_index]}_step_{step_idx}.png")
            # Image.fromarray(image_tensor).save(f"{sample_folder_dir}/{index:06d}_mask.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--config-file", type=str, default="configs/refcoco/val.json")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-My/2")
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=224)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--test-sample-num", type=int, default=512)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default='results/002-DiT-My-2-timestep100/checkpoints/0094200.pt',
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)