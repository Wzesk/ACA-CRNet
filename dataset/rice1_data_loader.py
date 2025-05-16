# -*- coding: utf-8 -*-
"""
Created on May 9 10:24:49 2024

@author: Wenli Huang
"""

from PIL import Image, ImageFile
import torchvision.transforms as transforms
import torch.utils.data as data

import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")

from image_folder import make_dataset
import os.path
import numpy as np

class CreateDataset(data.Dataset):
    def __init__(self,img_file, isTrain=False,load_size=256,crop_size=128):
        self.isTrain = isTrain
        self.load_size= load_size
        self.crop_size= crop_size
        self.img_paths, self.img_size = make_dataset(img_file)
        # provides random file for training and testing
        # if opt.mask_file != 'none':
        #     self.mask_paths, self.mask_size = make_dataset(opt.mask_file)
        #     if not self.opt.isTrain:
        #         self.mask_paths = self.mask_paths * (max(1, math.ceil(self.img_size / self.mask_size)))
        self.transform = get_transform(isTrain = isTrain, load_size=256,crop_size=128)

    def __getitem__(self, index):
        # load image
        img, img_path = self.load_img(index)
        img_gt, img_path_gt = self.load_img_gt(index)

        if self.load_size - self.crop_size > 0:
            if self.isTrain:
                y = np.random.randint(0, np.maximum(0, self.load_size - self.crop_size))
                x = np.random.randint(0, np.maximum(0, self.load_size - self.crop_size))
            else:
                y = np.maximum(0, self.load_size - self.crop_size)//2
                x = np.maximum(0, self.load_size - self.crop_size)//2
            img = img[...,y:y+self.crop_size,x:x+self.crop_size]
            img_gt = img_gt[...,y:y+self.crop_size,x:x+self.crop_size]
            #s2cldimg = s2cldimg[...,y:y+self.crop_size,x:x+self.crop_size]

        # load mask
        #mask = self.load_mask(img, index)
        return img,img_gt, img_path

    def __len__(self):
        return self.img_size

    def name(self):
        return "inpainting dataset"

    def load_img(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_path = self.img_paths[index % self.img_size]
        img_pil = Image.open(img_path).convert('RGB')
        img = self.transform(img_pil)
        img_pil.close()
        return img, img_path

    def load_img_gt(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_path = self.img_paths[index % self.img_size]

        line = img_path.strip('\n')
        pic = line.split('/')
        pic_des = pic[-1]
        pic_class = pic[-2]

        img_path_gt = ''#'/'
        for pa in pic[:-2]:
            img_path_gt = os.path.join(img_path_gt, pa)

        img_path_gt = img_path_gt + '/label/' + pic_des
        img_pil = Image.open(img_path_gt).convert('RGB')
        img = self.transform(img_pil)
        img_pil.close()
        return img, img_path_gt

    def load_mask(self, img, index):
        """Load different mask types for training and testing"""
        mask_type = 5

        if mask_type == 5:
            # if self.opt.isTrain:
            #     mask_index = random.randint(0, self.mask_size-1)
            # else:
            mask_index = index
            mask_pil = Image.open(self.mask_paths[mask_index]).convert('L')

            mask_transform = transforms.Compose([
                                                 transforms.Resize([self.load_size,self.load_size]),
                                                 transforms.ToTensor()
                                                 ])
            mask = (mask_transform(mask_pil) == 0).float()
            mask_pil.close()
            return mask


def dataloader(img_file,isTrain=False,batch_size=1,shuffle=False,nThreads=1,load_size=256,crop_size=128):
    datasets = CreateDataset(img_file, isTrain=isTrain,load_size=load_size,crop_size=crop_size)
    dataset = data.DataLoader(datasets, batch_size=batch_size, shuffle=shuffle, num_workers=int(nThreads))

    return dataset


def get_transform(isTrain = False, load_size=256,crop_size=128):
    """Basic process to transform PIL image to torch tensor"""
    transform_list = []
    osize = [load_size, load_size]
    fsize = [crop_size, crop_size]
    if isTrain:
        transform_list.append(transforms.Resize(osize))

    else:
        transform_list.append(transforms.Resize(osize))
        #transform_list.append(transforms.CenterCrop(fsize))

    transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)
