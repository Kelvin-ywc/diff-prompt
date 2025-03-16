import torch
import torch.nn as nn

import json

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

import pycocotools
from pycocotools import mask as maskUtils
from PIL import Image
import pytorch_lightning as L

class Cfg():
    # annotation_files = "D:\data\coco\annotations\refcoco-unc\instances.json"
    annotation_files = './asset/instances.json'
    train_annotation_files = [annotation_files, ]
    train_dataset_types = ['train']
    test_annotation_files = [annotation_files, annotation_files]
    test_dataset_types = ['testA', 'testB']
    val_annotation_files = [annotation_files, ]
    val_dataset_types = ['val']

class SegmentMaskData(Dataset):
    def __init__(self, annotation_files: str, mask_size: int=256, dataset_types=['train']):
        self.mask_transform = transforms.Compose([
            transforms.Resize([mask_size, mask_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        annotation = None
        for ann in annotation_files:
            for dataset_type in dataset_types:
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
        mask = self.annToMask(mask_rle, img_org_shape[1], img_org_shape[0])
        mask = Image.fromarray(mask)
        mask = self.mask_transform(mask)
        # mask*=255
        
        return mask
        
    def __len__(self):
        return len(self.masks)
    

class SegmentMaskModule(L.LightningDataModule):
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
        if stage == 'fit':
            self.train_dataset = SegmentMaskData(self.train_annotation_files)
        if stage == 'test':
            self.test_dataset = SegmentMaskData(self.test_annotation_files)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)


    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)