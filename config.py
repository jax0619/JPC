import os
from easydict import EasyDict as edict
import time
import torch
from utils.collections import AttrDict

__C = edict()
cfg = __C

__C.seed = 2453  # 2453  #3035 random seed, for reproduction

# ---------------the parameters of training----------------
__C.sou_dataset = 'GCC'
__C.model_type = 'vgg16'
__C.phase = 'train'  # choices=['train','test','pre_train','cross_train'])
__C.gpu_id = "0"  
__C.target_dataset = 'SHHA'#'QNRF' dataset choices =  ['SHHB', 'QNRF', 'MALL', 'SHHA']
__C.train_mode = 'train_few_shot'

__C.csrnet = False  # use CSRnet model
__C.pre =  False  #Image Net pre-trained weight

#data_split
__C.ratio=0.1 if __C.train_mode=='train_few_shot' else 1.
__C.split=False
__C.val=0.

__C.vis = False  # visualize data

__C.init_weights ='./'#pre-trained weight
__C.test_weight = './'#adapted weight

#perturbation
#feature-level
__C.TrainwF=True
__C.feature =False
__C.feature_noise = True
__C.dropout = 0.1
#image-level
__C.RandGaussianBlur = 1.
__C.MaskRandJitter = [0.9,0.8,0.5,0.1]
__C.size_scale = True
__C.train_size=(480,480)

__C.GCC_efficient = 0.3
__C.th_efficient = 0.5

__C.max_epoch = 400
__C.test_start_epoch = 150

#batchsize
__C.GCC_batch_size = 5
__C.real_batch_size = 5
__C.pre_train_batchsize = 12
__C.pre_test_batchsize = 12

#pre-training
__C.pre_lr = 1e-5
__C.pre_weight_decay = 1e-4
__C.pre_gamma = 0.98
__C.pre_step_size = 2
#DA
__C.da_lr = 1e-5
__C.da_weight_decay = 1e-4
__C.da_gamma = 0.98
__C.da_step_size = 2

__C.MODEL = AttrDict()
# momentum for the moving class prior
__C.MODEL.NET_MOMENTUM = 0.7 #0.7 or 0.99 retention ratio
__C.MODEL.NET_MOMENTUM_ITER = 1

# ------------------------------VIS------------------------
now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.EXP_NAME = now + "_" + __C.target_dataset + "_" + __C.model_type + "_" + \
               '_lr' + str(__C.da_lr) + '_' + \
               __C.phase + '_' + __C.train_mode

if __C.target_dataset == "SHHA":
    __C.EXP_PATH = "./exp/SHHA"
if __C.target_dataset == "QNRF":
    __C.EXP_PATH = "./exp/QNRF"
if __C.target_dataset == "SHHB":
    __C.EXP_PATH = "./exp/SHHB"
if __C.target_dataset == "MALL":
    __C.EXP_PATH = "./exp/MALL"
if __C.target_dataset == "UCSD":
    __C.EXP_PATH = "./exp/UCSD"
if __C.target_dataset == "UCF50":
    __C.EXP_PATH = "./exp/UCF50"
if __C.target_dataset == "GCC":
    __C.EXP_PATH = "./exp"
if __C.target_dataset == "WE":
    __C.EXP_PATH = "./exp/WE"
