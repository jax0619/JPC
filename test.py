##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Trainer for pretrain phase. """
import os.path as osp
import os
import tqdm
import numpy as np
import torch.nn.functional as F
from models.counter import NLT_Counter
from models.csrnet import CSRNet
from misc.utils import *
from tensorboardX import SummaryWriter
from dataloader.loading_data import loading_data
from dataloader.setting import cfg_data
import torchvision.utils as vutils
import torchvision.transforms as standard_transforms
from config import cfg
from models import get_model
from matplotlib import pyplot as plt
from itertools import cycle
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import sys
from collections import OrderedDict

class DATrainer(object):
    """The class that contains the code for the pretrain phase."""

    def __init__(self, cfg, gpu, pwd):
        # Set the folder to save trecords and checkpoints
        # Set cfg to be shareable in the class
        self.cfg_data = cfg_data
        self.pwd = pwd
        # self.exp_path = cfg.EXP_PATH
        # self.exp_name = cfg.EXP_NAME
        # self.exp_path = osp.join(self.exp_path, 'da')
        self.real_test_loader = loading_data(
            cfg)

        self.model = CSRNet() if cfg.csrnet else NLT_Counter()
        self.model.to(device='cuda')
        self.pretrained_dict = torch.load(cfg.init_weights)  # ['params']
        self.model.load_state_dict(self.pretrained_dict)

        self.record = {}

        self.record['val_loss'] = []
        self.record['val_mae'] = []
        self.record['val_mse'] = []

        self.record['test_loss'] = []
        self.record['test_mae'] = []
        self.record['test_mse'] = []

    def train(self):
        """The function for the pre_train on GCC dataset."""
        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0
        self.model.eval()

        loss_avg = Averager()
        mae_avg = Averager()
        mse_avg = Averager()

        # Run validation
        for i,(data,label) in enumerate(self.real_test_loader, 1):
            # print(i)
            with torch.no_grad():
                data = data.cuda()
                label = label.cuda()
                pred = self.model(data)
                loss = F.mse_loss(pred.squeeze(1), label)
                loss_avg.add(loss.item())

                # dist=EuclideanDistances(stu_pred[1].squeeze(),mmt_pred[1].squeeze())
                # dist=dist.sum()
                for img in range(pred.size()[0]):
                    pred_cnt = (pred[img] / self.cfg_data.LOG_PARA).sum().data
                    gt_cnt = (label[img] / self.cfg_data.LOG_PARA).sum().data
                    mae = torch.abs(pred_cnt - gt_cnt).item()
                    mse = (pred_cnt - gt_cnt).pow(2).item()
                    mae_avg.add(mae)
                    mse_avg.add(mse)


        # Update validation averagers
        loss_avg = loss_avg.item()
        mae_avg = mae_avg.item()
        mse_avg = np.sqrt(mse_avg.item())


        # Print loss and maeuracy for this epoch
        print('test, Loss={:.4f} mae={:.4f}  mse={:.4f}'.format(loss_avg, mae_avg,
                                                                         mse_avg))

if __name__ == '__main__':
    import torch

    pwd = os.path.split(os.path.realpath(__file__))[0]
    trainer = DATrainer(cfg, torch.cuda.device_count(), pwd)
    trainer.train()


