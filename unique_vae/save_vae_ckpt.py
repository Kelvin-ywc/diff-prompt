from unique_vae.experiment import VAEXperiment
from unique_vae.models import MaskVanillaVAE_V2

import yaml
import torch
with open('./configs/mask_vae_v2.yaml', 'r') as file:
    try:
        vae_config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
vae = MaskVanillaVAE_V2()
checkpoint_path = '/home1/yanweicai/workspace/prompt/PromptDiffusion/unique_vae/logs/MaskVanillaVAE_V2/version_20/checkpoints/epoch=157-step=13113.ckpt'
vae_experiment = VAEXperiment.load_from_checkpoint(vae_model=vae, params=vae_config['exp_params'], checkpoint_path=checkpoint_path)
# torch.save(vae.vae.state_dict(), '../ckpt/mask_vae_epoch=157-step=13113.bin')
# torch.save(vae.vae.state_dict(), '../ckpt/mask_vae_epoch=157-step=13113.safetensors')
vae.vae.save_pretrained('./ckpt/mask_vae_epoch=157-step=13113.safetensors',safe_serialization=True)

# vae.to(torch.device(0))

# print(1)