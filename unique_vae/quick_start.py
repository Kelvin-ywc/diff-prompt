from diffusers import AutoencoderKL
from dataset import SegmentMaskData
from torch.utils.data import DataLoader
import torch
from PIL import Image
from torchvision import transforms

def mask_to_image(mask: torch.Tensor) -> Image:
    mask = torch.clamp(255.0 * mask, 0, 255).detach().squeeze(dim=0).squeeze(0).to("cpu", dtype=torch.uint8).numpy()
    mask = Image.fromarray(mask)
    return mask

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load the pre-trained model
    vae = AutoencoderKL.from_pretrained('oaaoaa/mask_vae').to(device)
    print(vae)
    # load the mask image
    image = Image.open('./asset/test_mask.png')
    mask_size = 224

    preprocessor = transforms.Compose([
        transforms.Resize([mask_size, mask_size]),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = preprocessor(image).unsqueeze(0).to(device)
    scaling_factor = 0.18215
    # encode
    with torch.no_grad():
        latents = vae.encode(img_tensor).latent_dist.sample() * scaling_factor
    print(f'shape of latent feature: {latents.shape}')
    # decode
    with torch.no_grad():
        reconstucted = vae.decode(latents/ scaling_factor).sample
    reconstucted = mask_to_image(reconstucted)
    reconstucted.save('./asset/reconstructed_mask.png')

if __name__ == '__main__':
    main()


