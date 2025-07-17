from .dit_models import DiT_models
import torch


def load_dit(model_type, path, latent_size):
    try:
        local_rank = torch.distributed.get_rank()
    except:
        local_rank = 0
    checkpoint = torch.load(path, map_location='cuda:{}'.format(local_rank))['model']
    dit_model = DiT_models[model_type](input_size=latent_size, num_classes=1000)
    dit_model.load_state_dict(checkpoint)
    return dit_model