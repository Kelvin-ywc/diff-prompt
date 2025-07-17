import os
import json
from collections import defaultdict
from PIL import Image
import re

import torch
from torchvision import transforms
from torch.utils.data import Dataset

import pycocotools
from pycocotools import mask as maskUtils
from loguru import logger

from utils import data
from utils.data.transforms.transforms import build_trans, build_mask_trans

class Coco(Dataset):
    def __init__(self, cfg=None, is_train=True, dataset_type='train', max_words=30, task_type= 'segmentation',transform=None):
        # self.root = root
        # self.ann_file = ann_file
        self.root = cfg.data_path
        self.ann_file = cfg.annotation_path
        self.task_type = task_type
        self.mask_transform = build_mask_trans(is_train, cfg.image_size)
        if transform == None:
            self.transform = build_trans(is_train, cfg.image_size)
        else:
            self.transform = transform
        self.annotation = None
        for ann in self.ann_file:
            if self.annotation == None:
                self.annotation = json.load(open(ann, 'r'))[dataset_type]
            else:
                self.annotation.extend(json.load(open(ann, 'r'))[dataset_type])
            # if is_train:
            #     if self.annotation == None:
            #         self.annotation = json.load(open(ann, 'r'))['train']
            #     else:
            #         self.annotation.extend(json.load(open(ann, 'r'))['train'])
            #     # self.annotation.append(json.load(open(self.ann_file, 'r')))
            # elif dataset_type == 'testA':
            #     if self.annotation == None:
            #         self.annotation = json.load(open(ann, 'r'))['testA']
            #     else:
            #         self.annotation.extend(json.load(open(ann, 'r'))['testA'])
            # elif dataset_type == 'testB':
            #     if self.annotation == None:
            #         self.annotation = json.load(open(ann, 'r'))['testB']
            #     else:
            #         self.annotation.extend(json.load(open(ann, 'r'))['testB'])
            # elif dataset_type == 'val':
            #     if self.annotation == None:
            #         self.annotation = json.load(open(ann, 'r'))['val']
            #     else:
            #         self.annotation.extend(json.load(open(ann, 'r'))['val'])
        self.is_train = is_train

        self.dataset_type = dataset_type

        self.ids = []
        self.img_ids = []
        self.id2img = {}
        self.id2caption = {}
        self.id2mask = {}
        self.id2bbx = {}
        cur_id = 0
        logger.debug("Creating Index ....")
        # if is_train:
        #     self.annotation = self.annotation['train']
        # else:
        #     if self.dataset_type == 'testA':
        #         self.annotation = self.annotation['testA']
        #     elif self.dataset_type == 'testB':
        #         self.annotation = self.annotation['testB']
        #     elif self.dataset_type == 'val':
        #         self.annotation = self.annotation['val']
        #     elif self.dataset_type == 'all':
        #         self.annotation = self.annotation['testA'] + self.annotation['testB'] + self.annotation['val']

        for img_id, itm in enumerate(self.annotation):
            self.img_ids.append(img_id)
            for caption_id in range(len(itm['expressions'])):
                self.ids.append(cur_id)
                self.id2img[cur_id] = itm['image_id']
                self.id2caption[cur_id] = itm['expressions'][caption_id]
                self.id2bbx[cur_id] = itm['bbox']
                self.id2mask[cur_id] = itm['mask']
                cur_id += 1
        
        if (not is_train) and cfg.test_sample_num < len(self.ids):
            self.ids = self.ids[:cfg.test_sample_num]
        # FIXME
        self.ids = self.ids[:256]

        if is_train:
            logger.debug("Total training samples: {}".format(len(self.ids)))
        else:
            logger.debug("Total testA/ testB/ val samples: {}".format(len(self.ids)))
        self.max_words = max_words

    def __getitem__(self, idx):
        id = self.ids[idx]
        image_path = os.path.join(self.root, 'train2014/COCO_train2014_{:0>12d}.jpg'.format(self.id2img[id]))
        # Load image and annotations
        img = Image.open(image_path).convert('RGB')
        img_org_shape = img.size
        caption = pre_caption(self.id2caption[id], self.max_words)
        target = None
        if self.task_type == 'segmentation':
            mask_rle = self.id2mask[id] # rle
            mask = self.annToMask(mask_rle, img_org_shape[1], img_org_shape[0])
            # mask*=255
            # mask = Image.fromarray(mask).convert('RGB')
            mask = Image.fromarray(mask*255)

        elif self.task_type == 'detection':
            target = self.id2bbx[id]
            #TODO bbx -> mask   
            exit(0)
        #TODO save img, mask
        # logger.debug(f"Saving img and mask, {caption}")
        # img.save(f'./VIS/img_rog_{idx}.jpg')
        # mask.save(f'./VIS/mask_org.jpg')

        if self.transform:
            img = self.transform(img) # tensor
            mask = self.mask_transform(mask) # tensor

            #FIXME visualize the image and mask
            # toPIL = transforms.ToPILImage() #这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
            # mask = mask*255
            # img = toPIL(img)
            # mask = toPIL(mask)
            # img.save(f'./VIS/img.jpg')
            # mask.save(f'./VIS/mask.jpg')

            img = ImageObj(img, img_org_shape)
            mask = ImageObj(mask, img_org_shape)
        return img, mask, caption 

    def __len__(self):
        return len(self.ids)

    def annToMask(self, mask, h, w):
        rles = maskUtils.frPyObjects(mask, h, w)
        rle = maskUtils.merge(rles)
        m = maskUtils.decode(rle)
        return m
        

    

def pre_caption(caption,max_words=50):
	# 把这些符号：.!\"()*#:;~ 替换为空格，并且将caption全部转换为小写字母
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    # 将连续出现两个或更多空格的地方替换为单个空格
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    # 去掉caption末尾的换行符
    caption = caption.rstrip('\n') 
    # 去掉caption 两边的空格
    caption = caption.strip(' ')
    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words: # 如果超过了max_words，就只取前max_words个单词
        caption = ' '.join(caption_words[:max_words])
            
    return caption


class ImageObj():
    def __init__(self, image, org_shape) -> None:
        self.image = image
        self.org_shape = org_shape
        self.cur_shape = image.shape