from transformers import PretrainedConfig

class MaskDiTConfig(PretrainedConfig):
    model_type = "mask_dit"

# depth=12, hidden_size=512, patch_size=2, num_heads=8,    input_size=latent_size, num_classes=1000
    def __init__(self, depth=12, hidden_size=512, patch_size=2, num_heads=8, input_size=28, num_classes=1000, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.input_size = input_size
        self.num_classes = num_classes