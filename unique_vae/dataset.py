import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile
import pycocotools
from pycocotools import mask as maskUtils
from PIL import Image
import json
import numpy

# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass
    
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True
    
    

class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"        
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy datat to prevent breaking 

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
#       =========================  OxfordPets Dataset  =========================
            
#         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                               transforms.CenterCrop(self.patch_size),
# #                                               transforms.Resize(self.patch_size),
#                                               transforms.ToTensor(),
#                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
#         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                             transforms.CenterCrop(self.patch_size),
# #                                             transforms.Resize(self.patch_size),
#                                             transforms.ToTensor(),
#                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#         self.train_dataset = OxfordPets(
#             self.data_dir,
#             split='train',
#             transform=train_transforms,
#         )
        
#         self.val_dataset = OxfordPets(
#             self.data_dir,
#             split='val',
#             transform=val_transforms,
#         )
        
#       =========================  CelebA Dataset  =========================
    
        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.CenterCrop(148),
                                              transforms.Resize(self.patch_size),
                                              transforms.ToTensor(),])
        
        val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.patch_size),
                                            transforms.ToTensor(),])
        
        self.train_dataset = MyCelebA(
            self.data_dir,
            split='train',
            transform=train_transforms,
            download=False,
        )
        
        # Replace CelebA with your dataset
        self.val_dataset = MyCelebA(
            self.data_dir,
            split='test',
            transform=val_transforms,
            download=False,
        )
#       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
     
class Cfg():
    annotation_files = "/home1/yanweicai/DATA/CV/coco/annotations/refcoco-unc/instances.json"
    # annotation_files = './asset/instances.json'
    train_annotation_files = [annotation_files, ]
    train_dataset_types = ['train']
    test_annotation_files = [annotation_files, annotation_files]
    test_dataset_types = ['testA', 'testB']
    val_annotation_files = [annotation_files, ]
    val_dataset_types = ['val']

class SegmentMaskData(Dataset):
    def __init__(self, annotation_files: str, dataset_types=['train'], mask_size: int=224):
        self.mask_transform = transforms.Compose([
            transforms.Resize([mask_size, mask_size]),
            transforms.ToTensor(),
            # transforms.Normalize((0.00089937, ), (0.00247348, ))
            # transforms.Normalize((0.11487043, ), (0.31592062, ))
        ])
        annotation = None
        for ann, dataset_type in zip(annotation_files, dataset_types):
            if annotation == None:
                annotation = json.load(open(ann, 'r'))[dataset_type]
            else:
                annotation.extend(json.load(open(ann, 'r'))[dataset_type])
        self.masks = []
        self.org_shapes = []
        for ann in annotation:
            self.masks.append(ann['mask'])
            self.org_shapes.append([ann['height'], ann['width']])
            
    def annToMask(self, mask, h, w):
        rles = maskUtils.frPyObjects(mask, h, w)
        rle = maskUtils.merge(rles)
        m = maskUtils.decode(rle)
        return m
    
    def __getitem__(self, idx):
        mask_rle = self.masks[idx]
        img_org_shape = self.org_shapes[idx]
        mask = self.annToMask(mask_rle, img_org_shape[0], img_org_shape[1]) # 0,1
        # Image.fromarray(mask*255).save(f"PyTorch-VAE/vis/mask_org_1_{idx}.png")

        mask = Image.fromarray(mask*255)
        
        mask = self.mask_transform(mask)
        # mask*=255
        
        return mask, 0.0
        
    def __len__(self):
        return len(self.masks)
    
class SegmentMaskModule(LightningDataModule):
    def __init__(self, cfg: Cfg, batch_size: int = 32):
        super().__init__()
        if hasattr(cfg, 'train_annotation_files'):
            self.train_annotation_files = cfg.train_annotation_files
            self.train_dataset_types = cfg.train_dataset_types
        if hasattr(cfg, 'val_annotation_files'):
            self.val_annotation_files = cfg.val_annotation_files
            self.val_dataset_types = cfg.val_dataset_types
        if hasattr(cfg, 'test_annotation_files'):
            self.test_annotation_files = cfg.test_annotation_files
            self.test_dataset_types = cfg.test_dataset_types
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.train_dataset = SegmentMaskData(self.train_annotation_files, self.train_dataset_types)
        self.val_dataset = SegmentMaskData(self.val_annotation_files, self.val_dataset_types)
        self.test_dataset = SegmentMaskData(self.test_annotation_files, self.test_dataset_types)
        


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)