import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataloader.GCC import  GCC
from dataloader.MALL import  MALL
from dataloader.SHHB import SHHB
from dataloader.SHHA import SHHA
from dataloader.QNRF import QNRF
from dataloader.setting import cfg_data
import torchvision
import torch
import random

def loading_data(args):
    mean_std = cfg_data.MEAN_STD
    log_para = cfg_data.LOG_PARA

    sou_main_transform = own_transforms.Compose([
        own_transforms.RandGaussianBlur(),
        own_transforms.MaskRandJitter(),
    ]) if args.phase!='pre_train' else None


    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    gt_transform = standard_transforms.Compose([
        own_transforms.LabelNormalize(log_para)
    ])
    if args.phase=='pre_train' or args.phase=='SE_pre_train':
        GCCbatchsize=args.pre_train_batchsize
    else:
        GCCbatchsize=args.GCC_batch_size
    GCC_trainset = GCC('train', main_transform=sou_main_transform, img_transform=img_transform,
                       gt_transform=gt_transform)
    GCC_train_loader = DataLoader(GCC_trainset, batch_size=GCCbatchsize, shuffle=True, num_workers=4,
                                  drop_last=True, pin_memory=True)
    GCC_testset = GCC('test', img_transform=img_transform, gt_transform=gt_transform)
    GCC_test_loader = DataLoader(GCC_testset, batch_size=args.pre_test_batchsize, shuffle=True, num_workers=4,
                                 drop_last=True, pin_memory=True)

    if args.target_dataset == 'SHHA' or args.target_dataset =='QNRF':
        batchsize=1
        num_workers=1
    else:
        batchsize=8
        num_workers=4

    if args.phase == 'cross_train':
        real_trainset = SHHB(args.train_mode, main_transform=sou_main_transform, img_transform=img_transform,
                             gt_transform=gt_transform)
        real_testset = MALL('test', img_transform=img_transform, gt_transform=gt_transform)
    else:
        if args.target_dataset == 'SHHA':
            real_trainset = SHHA(args.train_mode, main_transform=sou_main_transform, img_transform=img_transform,gt_transform=gt_transform)
            real_testset = SHHA('test', img_transform=img_transform, gt_transform=gt_transform)
        elif args.target_dataset == 'SHHB':
            real_trainset = SHHB(args.train_mode, main_transform=sou_main_transform, img_transform=img_transform,gt_transform=gt_transform)
            real_testset = SHHB('test', img_transform=img_transform, gt_transform=gt_transform)
        elif args.target_dataset == 'QNRF':
            real_trainset = QNRF(args.train_mode, main_transform=sou_main_transform, img_transform=img_transform,gt_transform=gt_transform)
            real_testset = QNRF('test', img_transform=img_transform, gt_transform=gt_transform)
        elif args.target_dataset == 'MALL':
            real_trainset = MALL(args.train_mode, main_transform=sou_main_transform, img_transform=img_transform,gt_transform=gt_transform)
            real_testset = MALL('test', img_transform=img_transform, gt_transform=gt_transform)
    if args.target_dataset != 'GCC':
        real_train_loader = DataLoader(real_trainset, batch_size=args.real_batch_size, shuffle=True, num_workers=4,
                                   drop_last=True, pin_memory=True)
        real_test_loader = DataLoader(real_testset, batch_size=batchsize, num_workers=num_workers, pin_memory=True)

    if args.phase=='test':
        if args.target_dataset=='GCC':
            return GCC_test_loader
        else:
            return real_test_loader

    if args.phase == 'pre_train':
        if args.target_dataset=='GCC':
            return GCC_train_loader, GCC_test_loader
        else:
            return real_train_loader,real_test_loader

    if args.phase=='SE_pre_train':
        return GCC_train_loader, GCC_test_loader

    return GCC_train_loader,real_train_loader, real_test_loader




