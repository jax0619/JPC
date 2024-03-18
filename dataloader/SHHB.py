import numpy as np
import os
import random
from scipy import io as sio
import sys
import torch
from torch.utils import data
from PIL import Image, ImageOps

import pandas as pd
from  dataloader.setting import cfg_data
from misc.transforms import RandomCrop,Randomresize


class SHHB(data.Dataset):
    def __init__(self, mode, main_transform=None, img_transform=None, gt_transform=None):

        self.mode = mode
        # print("target"+mode+"path",data_path)

        with open(os.path.join(cfg_data.SHHB_scene_dir,mode,mode+'.txt')) as f:
            lines = f.readlines()

        self.data_files = []
        for line in lines:
            line = line.strip('\n')
            self.data_files.append(line)
        if self.mode == 'train':
            print('target ' + mode, len(self.data_files))
            self.data_files = int(cfg_data.num_batch * cfg_data.target_shot_size / (len(self.data_files))) * self.data_files
            print('target ' + mode + ' expand to', len(self.data_files))
        self.num_samples = len(self.data_files)
        self.op_crop = RandomCrop(cfg_data.TRAIN_SIZE)
        # self.op_crop = RandomCrop((1024, 1024))
        if self.mode is not 'train':
            print('target '+mode, self.num_samples)

        self.op_resize = Randomresize()
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform

    def __getitem__(self, index):
        fname = self.data_files[index]

        img, den = self.read_image_and_gt(fname)

        if self.main_transform is not None:
            ns = [0, 0]
            ns_img, ns_den, ns = self.main_transform(img, den, ns)
            # img, den = self.op_resize(img, den)
            # ns_img, ns_den, x, y = self.op_crop(ns_img, ns_den)
            # img, den, x0, y0 = self.op_crop(img, den, x1=x, y1=y)
            if self.img_transform is not None:
                img = self.img_transform(img)
                ns_img = self.img_transform(ns_img)
                # den = torch.from_numpy(np.array(den, dtype=np.float32))
            if self.gt_transform is not None:
                den = self.gt_transform(den)
                ns_den = self.gt_transform(ns_den)
            return img, ns_img, den, ns
        else:
            if self.img_transform is not None:
                img = self.img_transform(img)

                # den = torch.from_numpy(np.array(den, dtype=np.float32))
            if self.gt_transform is not None:
                den = self.gt_transform(den)
            return img, den


    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self, fname):
        if self.mode == 'train' or 'val':
            img_path = cfg_data.SHHB_DATA_PATH + '/train/img/' + fname + '.jpg'
            den_path = cfg_data.SHHB_DATA_PATH + '/train/den/' + fname + '.csv'
        if  self.mode == 'test':
            img_path = cfg_data.SHHB_DATA_PATH + '/test/img/' + fname + '.jpg'
            den_path = cfg_data.SHHB_DATA_PATH + '/test/den/' + fname + '.csv'

        img = Image.open(img_path)
        if img.mode == 'L':
            img = img.convert('RGB')

        # den = sio.loadmat(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.mat'))
        # den = den['map']
        den = pd.read_csv(den_path, sep=',', header=None).values

        den = den.astype(np.float32, copy=False)

        # print(den.sum(), os.path.join(os.path.splitext(fname)[0] + '.csv'))
        #
        den = Image.fromarray(den)



         #Creates an image memory from an object exporting the array interface (using the buffer protocol).\u4e0d\u6539\u53d8\u6570\u503c

        return img, den

    def get_num_samples(self):
        return self.num_samples

