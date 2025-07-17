from termios import CKILL
from models import DiT_models
import torch
from download import find_model
from transformers import BertConfig, BertModel
from mask_dit import MaskDiTConfig, MaskDiT
from huggingface_hub import login
from huggingface_hub import HfApi, HfFolder, Repository


# ===========CUSTOMIZE==============
ckpt_path = '/home1/yanweicai/workspace/prompt/PromptDiffusion/DiT-main/results/002-DiT-My-2-timestep100/checkpoints/0094200.pt'
repository_id = "oaaoaa/mask_dit"
HF_TOKEN = 'HF_TOKEN'
# ==================================

latent_size = 224 // 8
model = DiT_models['DiT-My/2'](
    input_size=latent_size,
    num_classes=1000
).to(torch.device(0))

state_dict = find_model(ckpt_path)
maskDiTConfig = MaskDiTConfig()
model = MaskDiT(maskDiTConfig)
model.load_state_dict(state_dict, strict=False)
model.save_pretrained('./ckpt/')
print('done')


login(HF_TOKEN, add_to_git_credential=True)
repo = Repository(local_dir="./ckpt", clone_from=repository_id)
repo.git_add()
repo.git_commit("Add model and tokenizer")
repo.git_push()