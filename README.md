# [ICLR 2025] Official implementation of the paper "Diff-Prompt: Diffusion-Driven Prompt Generator with Mask Supervision"
<!-- Code to be released soon. -->
## TODO
- [x] Release code for Stage one, training code for Mask-VAE.
- [x] Release code for Stage two, training code for Diff-Prompt.
- [ ] Release code for Stage three, integrating Diff-Prompt with GLIP.
## Stage one: training Mask-VAE
We use AutoencoderKL from diffusers for training Mask-VAE.
### Environment Setup
```
cd unique_vae
pip install -r requirements.txt
```
### Training Script
```
python run.py --config configs/mask_vae_v2.yaml
```
Set hyper parameters in the configs/mask_vae_v2.yaml.

[//]: # (### Submit the model to huggingface)

[//]: # (```)

[//]: # (python save_vae_ckpt.py)

[//]: # (```)


### Quick start
We provide pre-trained model [here](https://huggingface.co/oaaoaa/mask_vae), and a simple demo for using the Mask-VAE in quick_start.py file. Reconstruction results are saved in the `asset` folder.
```python
python quick_start.py
```

## Stage two: training Diff-Prompt
### Environment Setup
```
cd DiT-main
pip install -r requirements.txt
```
### Training Script
```
torchrun --nnodes=1 --nproc_per_node 4 train.py --global-batch-size 128
```
Model weights are saved in results dir.
### save model to huggingface
replace ckpt_path,repository_id,HF_TOKEN in this file.
```
python save_dit_model_safe.py
```
We provide pre-trained model [here](https://huggingface.co/oaaoaa/mask_dit).
## Stage three: integrating Diff-Prompt with GLIP

## Citation
If you use our work, please consider citing:
```bibtex
@inproceedings{yan2025diff,
  title={Diff-Prompt: Diffusion-driven Prompt Generator with Mask Supervision},
  author={Yan, Weicai and Lin, Wang and Guo, Zirun and Wang, Ye and Feng, Fangming and Yang, Xiaoda and Wang, Zehan and Jin, Tao},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```

## Acknowledgements

Our code is based on the following repositories. We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.

- https://github.com/AntixK/PyTorch-VAE
- https://github.com/facebookresearch/DiT
- https://github.com/microsoft/GLIP