
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Trainer for pretrain phase. """
import os.path as osp
import os
import tqdm
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from models.counter import Counter
from models.csrnet import CSRNet
from misc.utils import *
from misc.transforms import RandomHorizontallyFlip
from tensorboardX import SummaryWriter
from dataloader.loading_data import loading_data
from dataloader.setting import cfg_data
import torchvision.utils as vutils
import torchvision.transforms as standard_transforms
from config import cfg
from matplotlib import pyplot as plt
from itertools import cycle
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import sys
from data_split import data_split as ds
class DATrainer(object):
    """The class that contains the code for the pretrain phase."""
    def __init__(self, cfg,gpu,pwd):
        # Set the folder to save trecords and checkpoints
        # Set cfg to be shareable in the class
        self.cfg_data = cfg_data
        self.pwd = pwd
        self.exp_path = cfg.EXP_PATH
        self.exp_name = cfg.EXP_NAME
        self.exp_path = osp.join(self.exp_path, 'da')
        self.val_loader=None
        if cfg.split:
            ds.generate_split()
        # self.GCC_train_loader, self.real_train_loader,self.val_loader,self.real_test_loader,self.restore_transform = loading_data(cfg)
        self.GCC_train_loader, self.real_train_loader, self.real_test_loader = loading_data(cfg)
        self.student=CSRNet() if cfg.csrnet else Counter()
        self.teacher=CSRNet() if cfg.csrnet else Counter()
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        n=0
        self.pretrained_dict = torch.load(cfg.init_weights)

        # for i, b in self.pretrained_dict.items():
        #     print(i, b.shape)
        # for i, b in self.teacher.state_dict().items():
        #     print(i, b.shape)
        new_state_dict = OrderedDict()
        if not cfg.csrnet:
            for i, b in self.teacher.state_dict().items():
                print(i, b.shape)

            if cfg.init_weights is not None:
                self.pretrained_dict = torch.load(cfg.init_weights)
                for i, b in self.pretrained_dict.items():
                    print(i, b.shape)

                if cfg.TrainwF:
                    # w/ feature perturbation
                    for i, b in self.pretrained_dict.items():
                        t = i.split('.')[2]
                        if i.split('.')[0] == 'encoder':
                            if int(i.split('.')[2]) > 16:
                                t = str(int(i.split('.')[2]) - 3)
                            elif int(i.split('.')[2]) > 8:
                                t = str(int(i.split('.')[2]) - 2)
                            elif int(i.split('.')[2]) > 2:
                                t = str(int(i.split('.')[2]) - 1)

                            i = i.replace(i.split('.')[2], t)
                            new_state_dict[i] = b
                            # if i.split('.')[3] == 'bias':
                            #     n += 1
                        else:
                            new_state_dict[i] = b
                    self.student.load_state_dict(new_state_dict)
                    self.teacher.load_state_dict(new_state_dict)
                else:
                    # w/o feature perturbation
                    self.student.load_state_dict(self.pretrained_dict)
                    self.teacher.load_state_dict(self.pretrained_dict)

        else:
            if cfg.init_weights is not None:  # ['params']
                for i, b in self.pretrained_dict.items():
                    t = i.split('.')[1]
                    if i.split('.')[0] == 'frontend':
                        if int(i.split('.')[1]) > 16:
                            t = str(int(i.split('.')[1]) - 3)
                        elif int(i.split('.')[1]) > 8:
                            t = str(int(i.split('.')[1]) - 2)
                        elif int(i.split('.')[1]) > 2:
                            t = str(int(i.split('.')[1]) - 1)

                        i = i.replace(i.split('.')[1], t)
                        new_state_dict[i] = b
                    else:
                        new_state_dict[i] = b
                self.student.load_state_dict(new_state_dict)
                self.teacher.load_state_dict(new_state_dict)

        self.optimizer = torch.optim.Adam(self.student.parameters(), lr = cfg.da_lr,weight_decay=cfg.da_weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg.da_step_size,gamma=cfg.da_gamma)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
            self.student = torch.nn.DataParallel(self.student).cuda()
            self.teacher = torch.nn.DataParallel(self.teacher).cuda()
        # self.op_hp=RandomHorizontallyFlip()
        self.record = {}
        self.record['train_loss'] = []
        self.record['train_mae'] = []
        self.record['train_mse'] = []

        self.record['test_loss'] = []
        self.record['test_mae'] = []
        self.record['test_mse'] = []
        

        self.record['best_mae'] = 1e10
        self.record['best_mse'] = 1e10
        self.record['best_model_name'] =''

        self.record['update_flag'] = 0

        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, ["exp"])
    def save_model(self, name,stu):
        if stu:
            torch.save(dict(params=self.student.module.state_dict()), osp.join(self.exp_path, self.exp_name,name + '.pth'))
        else:
            torch.save(dict(params=self.teacher.module.state_dict()), osp.join(self.exp_path, self.exp_name,name + '.pth'))

    def train(self):
        """The function for the pre_train on GCC dataset."""
        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0

        stu_all_loss=[]
        stu_all_mae = []
        stu_all_mse = []

        th_all_loss = []
        th_all_mae = []
        th_all_mse = []

        for epoch in range(1, cfg.max_epoch + 1):
            self.student.train()
            self.teacher.eval()
            #分别初始化学生和教师模型记录
            stu_train_loss_avg = Averager()
            stu_train_mae_avg = Averager()
            stu_train_mse_avg = Averager()

            th_train_loss_avg = Averager()
            th_train_mae_avg = Averager()
            th_train_mse_avg = Averager()
                
            # 合并两个dataloader
            #loader=zip(cycle(self.real_train_loader),self.GCC_train_loader)
            loader = zip(self.real_train_loader, self.GCC_train_loader)
            train_bar = tqdm.tqdm(loader,  total=len(self.real_train_loader))

            # gcc,real= next(iter(loader))
            count=0
            for n, batch in enumerate(train_bar, 1):
                count+=1
                global_count = global_count + 1
                #GCC合成数据
                #real真实数据
                #batch0,1,2分别是原图、标签、加噪图

                GCC_img   = batch[1][0].cuda()
                GCC_label = batch[1][2].cuda()

                size_scale=feature_noise=False
                real_img = batch[0][0].cuda()
                real_label = batch[0][2].cuda()
                real_ns_img = batch[0][1].cuda()
                ns=batch[0][3]
                # noise=[ns[0][j].item()+ns[1][j].item() for j in range(len(ns[0]))]
                noise=ns[0].sum().item()+ns[1].sum().item()
                #real_ns_img,real_label,t=self.op_hp(real_ns_img,real_label)
                #real_ns_img = real_ns_img + torch.randn(real_ns_img.size()).cuda() * 0.1
                if noise<=cfg.GCC_batch_size:
                    if random.random() < 0.6:
                        size_scale=cfg.size_scale
                    else:
                        feature_noise=cfg.feature_noise
                if cfg.target_dataset=='MALL':
                    # ratio = random.randrange(500, 2000) / 1000.0
                    ratio = random.randrange(850, 1150) / 1000.0
                else:
                    #ratio = random.randrange(500, 2000) / 1000.0
                    ratio = random.randrange(717, 1414) / 1000.0
                #if mall (500,2000)
                #ratio = random.randrange(850, 1150) / 1000.0
                new_size = int(cfg_data.TRAIN_SIZE[0] * ratio / 8) * 8
                if cfg.target_dataset =='SHHA' or cfg.target_dataset=='QNRF':
                    randommax = torch.zeros((cfg.real_batch_size, 2), dtype=int)
                    I = torch.sum(real_img, 1)
                    x = torch.sum(I, 1)
                    y = torch.sum(I, 2)
                    for img in range(len(real_img)):
                        for i in range(x.size()[1]):
                            if x[img][i] == 0:
                                randommax[img][0] = i
                                break
                        if randommax[img][0] == 0:
                            randommax[img][0] = y.size()[1]
                        for j in range(y.size()[1]):
                            if y[img][j] == 0:
                                randommax[img][1] = j
                                break
                        if randommax[img][1] == 0:
                            randommax[img][1] = x.size()[1]

                    cropped_real_ns_img, xy = randomcrop(real_ns_img, randommax=randommax)
                    cropped_real_label, a = randomcrop(real_label, randomaxis=xy)
                    cropped_real_img, a = randomcrop(real_img, randomaxis=xy)
                    SHRF=True
                else:
                    x1 = random.randint(0, real_img.size()[2]-cfg_data.TRAIN_SIZE[0])
                    y1 = random.randint(0, real_img.size()[3]-cfg_data.TRAIN_SIZE[1])
                    cropped_real_ns_img=real_ns_img[:,:,x1:x1+cfg_data.TRAIN_SIZE[0],y1:y1+cfg_data.TRAIN_SIZE[1]]
                    cropped_real_label = real_label[ :, x1:x1 + cfg_data.TRAIN_SIZE[0],y1:y1 + cfg_data.TRAIN_SIZE[1]]
                    cropped_real_img = real_img[:, :, x1:x1 + cfg_data.TRAIN_SIZE[0],y1:y1 + cfg_data.TRAIN_SIZE[1]]
                    SHRF = False
                if size_scale:
                    cropped_real_ns_img = F.interpolate(cropped_real_ns_img, size=(new_size, new_size), mode='bilinear',
                                                    align_corners=False)
                if cfg.vis:
                    vis_img(cropped_real_img, 'cropped_real_img',count)
                    vis_img(cropped_real_label.squeeze(), 'cropped_real_label',count,True)
                    vis_img(cropped_real_ns_img, 'cropped_real_ns_img',count)
                    vis_img(real_ns_img, 'real_ns_img',count)
                    vis_img(real_label.squeeze(), 'real_label',count,True)
                    vis_img(real_img, 'real_img',count)
                    vis_img(GCC_img, 'GCC_img',count)
                    vis_img(GCC_label.squeeze(),'GCC_label',count,True)

                pred1 = self.student(GCC_img)
                pred2 = self.student(cropped_real_img)

                if cfg.vis:vis_img(pred1.detach().squeeze(1),'pred1',count,True)
                if cfg.vis:
                    vis_img(pred2.detach().squeeze(1),'pred2',count,True)

                loss1= F.mse_loss(pred1.squeeze(1), GCC_label)*cfg.GCC_efficient

                loss1+=F.mse_loss(pred2.squeeze(1), cropped_real_label)

                self.optimizer.zero_grad()
                loss1.backward()
                self.optimizer.step()

                update_teacher = n % cfg.MODEL.NET_MOMENTUM_ITER == 0
                if update_teacher:
                    self.momentum_update(cfg.MODEL.NET_MOMENTUM)

                pred_stu = self.student(cropped_real_ns_img, noise=feature_noise)
                if cfg.vis:vis_img(pred_stu.detach().squeeze(1),'stu_pred',count,True)

                if size_scale:
                    pred_stu=F.interpolate(pred_stu,size=cfg_data.TRAIN_SIZE,mode='bilinear',align_corners=False)
                if cfg.vis:vis_img(pred_stu.detach().squeeze(1), 'ori_stu_pred',count,True)
                pred_th = self.teacher(real_img) if size_scale else self.teacher(cropped_real_img)
                if cfg.vis:vis_img(pred_th.detach().squeeze(1),'th_pred',count,True)
                #size_scale consistency
                if size_scale:
                    if SHRF:
                        pred_th, a = randomcrop(pred_th.squeeze(1), randomaxis=xy)
                    else:
                        pred_th=pred_th[:,:,x1:x1+cfg_data.TRAIN_SIZE[0],y1:y1+cfg_data.TRAIN_SIZE[1]]
                    if cfg.vis:vis_img(pred_th.detach().squeeze(1), 'cropped_th_pred',count,True)
                #vis_img(real_pred["th_pred"].squeeze(), 'cropped_th_pred')
                # GCC_losses,GCC_pred = self.model(GCC_ns_img,y=GCC_label,x2=GCC_img,use_teacher=True,update_teacher=update_teacher)
                #real_losses, real_pred = self.model(real_ns_img, y=real_label, x2=real_img, use_teacher=True,update_teacher=update_teacher)

                real_stu_loss = F.mse_loss(pred_stu.squeeze(),pred_th.squeeze())

                real_th_loss = F.mse_loss(pred_th.squeeze(),cropped_real_label.squeeze())*cfg.th_efficient

                Loss = real_stu_loss+real_th_loss

                self.optimizer.zero_grad()
                Loss.backward()
                self.optimizer.step()

                real_label_cnt = cropped_real_label.sum().data / self.cfg_data.LOG_PARA
                real_stu_pred_cnt = pred_stu.sum().data / self.cfg_data.LOG_PARA
                real_th_pred_cnt = pred_th.sum().data / self.cfg_data.LOG_PARA

                # GCC_stu_mae = torch.abs(GCC_label_cnt-GCC_stu_pred_cnt).item()
                # GCC_stu_mse = (GCC_label_cnt - GCC_stu_pred_cnt).pow(2).item()

                real_stu_mae = torch.abs(real_label_cnt - real_stu_pred_cnt).item()
                real_stu_mse = (real_label_cnt - real_stu_pred_cnt).pow(2).item()

                # GCC_th_mae = torch.abs(GCC_label_cnt - GCC_stu_pred_cnt).item()
                # GCC_th_mse = (GCC_label_cnt - GCC_stu_pred_cnt).pow(2).item()

                real_th_mae = torch.abs(real_label_cnt - real_th_pred_cnt).item()
                real_th_mse = (real_label_cnt - real_th_pred_cnt).pow(2).item()

                train_bar.set_description('Epoch {}, real_stu_loss={:.4f} gt={:.1f} real_stu_pred={:.1f} real_th_pred={:.1f} lr={:.4f} '.format(epoch, real_stu_loss.item(),real_label_cnt ,real_stu_pred_cnt,real_th_pred_cnt, self.optimizer.param_groups[0]['lr']*10000))
            #     # Add loss and maeuracy for the averagers
                stu_train_loss_avg.add(real_stu_loss.item())
                stu_train_mae_avg.add(real_stu_mae)
                stu_train_mse_avg.add(real_stu_mse)

                # if cfg.supervised:th_train_loss_avg.add(real_th_loss.item())
                th_train_mae_avg.add(real_th_mae)
                th_train_mse_avg.add(real_th_mse)

            # Update the averagers########
            stu_train_loss_avg = stu_train_loss_avg.item()
            stu_train_mae_avg = stu_train_mae_avg.item()
            stu_train_mse_avg = np.sqrt(stu_train_mse_avg.item())

            self.writer.add_scalar('data/loss',stu_train_loss_avg, global_count)
            self.writer.add_scalar('data/mae', stu_train_mae_avg, global_count)
            self.writer.add_scalar('data/mse', stu_train_mse_avg, global_count)
            # Start validation for this epoch, set model to eval mode

            self.student.eval()
            self.teacher.eval()

            stu_loss_avg = Averager()
            stu_mae_avg = Averager()
            stu_mse_avg = Averager()

            th_loss_avg = Averager()
            th_mae_avg = Averager()
            th_mse_avg = Averager()

            if epoch+1>cfg.test_start_epoch:
                # Run validation
                
                for i, batch in enumerate(self.real_test_loader, 1):
                    # print(i)
                    with torch.no_grad():
                        data = batch[0].cuda()
                        label = batch[1].cuda()
                        stu_pred = self.student(data)
                        th_pred=self.teacher(data)

                        stu_loss = F.mse_loss(stu_pred.squeeze(1), label)
                        th_loss = F.mse_loss(th_pred.squeeze(1), label)

                        stu_loss_avg.add(stu_loss.item())
                        th_loss_avg.add(th_loss.item())

                        for img in range(stu_pred.size()[0]):

                            stu_pred_cnt = (stu_pred[img] / self.cfg_data.LOG_PARA).sum().data
                            th_pred_cnt = (th_pred[img] / self.cfg_data.LOG_PARA).sum().data

                            gt_cnt = (label[img] / self.cfg_data.LOG_PARA).sum().data

                            stu_mae = torch.abs(stu_pred_cnt - gt_cnt).item()
                            stu_mse = (stu_pred_cnt - gt_cnt).pow(2).item()

                            th_mae = torch.abs(th_pred_cnt - gt_cnt).item()
                            th_mse = (th_pred_cnt - gt_cnt).pow(2).item()

                            stu_mae_avg.add(stu_mae)
                            stu_mse_avg.add(stu_mse)

                            th_mae_avg.add(th_mae)
                            th_mse_avg.add(th_mse)


                # Update validation averagers
                stu_loss_avg = stu_loss_avg.item()
                stu_mae_avg = stu_mae_avg.item()
                stu_mse_avg = np.sqrt(stu_mse_avg.item())

                th_loss_avg = th_loss_avg.item()
                th_mae_avg = th_mae_avg.item()
                th_mse_avg = np.sqrt(th_mse_avg.item())

                stu_all_loss.append(stu_loss_avg)
                stu_all_mae.append(stu_mae_avg)
                stu_all_mse.append(stu_mse_avg)

                th_all_loss.append(th_loss_avg)
                th_all_mae.append(th_mae_avg)
                th_all_mse.append(th_mse_avg)

                stu = False
                if stu_mae_avg<th_mae_avg or stu_mse_avg+5<th_mse_avg:
                    best_mae=stu_mae_avg
                    best_mse=stu_mse_avg
                    best_loss=stu_loss_avg
                    stu = True
                else:
                    best_mae = th_mae_avg
                    best_mse = th_mse_avg
                    best_loss = th_loss_avg

                self.writer.add_scalar('data/test_loss', float(best_loss), epoch)
                self.writer.add_scalar('data/test_mae', float(best_mae), epoch)
                self.writer.add_scalar('data/test_mse', float(best_mse), epoch)
                # Print loss and maeuracy for this epoch
                print('Epoch {}, test, Loss={:.4f} mae={:.4f}  mse={:.4f}   stu={}'.format(epoch, best_loss, best_mae,
                                                                                  best_mse,stu))

                # self.record['val_loss'].append(val_loss_avg)
                # self.record['val_mae'].append(val_mae_avg)

                self.record['test_loss'].append(best_loss)
                self.record['test_mae'].append(best_mae)
                self.record['test_mse'].append(best_mse)

                module=self.student.module if stu else self.teacher.module
                which='_stu'if stu else '_th'
                self.record = update_model(
                    which,module, epoch, self.exp_path, self.exp_name, [best_mae, best_mse, best_loss],
                    self.record,
                    self.log_txt)

                if epoch % 10 == 0:
                    print('Running Time: {}, Estimated Time: {}'.format(timer.measure(),
                                                                        timer.measure(epoch / cfg.max_epoch)))
                    # Save model every 10 epochs
                    print('Best Epoch {}, Best test mae={:.2f} mse={:.2f}'.format(self.record['best_model_name'],
                                                                                 self.record['best_mae'],
                                                                                 self.record['best_mse']))
                    self.save_model('epoch' + str(epoch) +'_'+ str(best_mae)+ '_' +which,stu)

        STU_Loss = torch.tensor(stu_all_loss)
        STU_Mae = torch.tensor(stu_all_mae)
        STU_Mse = torch.tensor(stu_all_mse)
        TH_Loss = torch.tensor(th_all_loss)
        TH_Mae = torch.tensor(th_all_mae)
        TH_Mse = torch.tensor(th_all_mse)
        if not os.path.exists(osp.join(self.exp_path, self.exp_name,'studata')):
            os.mkdir(osp.join(self.exp_path, self.exp_name,'studata'))
        if not os.path.exists(osp.join(self.exp_path, self.exp_name,'thdata')):
            os.mkdir(osp.join(self.exp_path, self.exp_name,'thdata'))

        torch.save(STU_Loss, osp.join(self.exp_path, self.exp_name,'studata', 'epoch_loss.pt'))
        torch.save(STU_Mae, osp.join(self.exp_path, self.exp_name,'studata', 'epoch_mae.pt'))
        torch.save(STU_Mse, osp.join(self.exp_path, self.exp_name,'studata', 'epoch_mes.pt'))

        torch.save(TH_Loss, osp.join(self.exp_path, self.exp_name,'thdata', 'epoch_loss.pt'))
        torch.save(TH_Mae, osp.join(self.exp_path, self.exp_name,'thdata', 'epoch_mae.pt'))
        torch.save(TH_Mse, osp.join(self.exp_path, self.exp_name,'thdata', 'epoch_mse.pt'))

        self.lr_scheduler.step()
        self.writer.close()
    def momentum_update(self,ratio):
        slow_net_dict = self.teacher.state_dict()
        backbone_dict = self.student.state_dict()
        for key, val in backbone_dict.items():
            if key.split(".")[-1] in ("weight", "bias", "running_mean", "running_var"):
                slow_net_dict[key].mul_(ratio)
                slow_net_dict[key].add_(val * (1. - ratio))


def vis_img(batch,name,order,label=False):
    batch=batch.cpu()
    if len(batch.size())==4:
        batch=batch.permute(0,2,3,1)
    for i in range(len(batch)):
        batch[i] = batch[i] / torch.max(batch[i])
    batch=batch*255
    path='./imgshow/'+str(order)+'/'+name
    if not os.path.exists(path):

        os.makedirs(path)


    for i in range(len(batch)):

        den = np.uint8(batch[i])

        den_frame = plt.gca()
        plt.imshow(den, cmap='jet')
        den_frame.axes.get_yaxis().set_visible(False)
        den_frame.axes.get_xaxis().set_visible(False)
        den_frame.spines['top'].set_visible(False)
        den_frame.spines['bottom'].set_visible(False)
        den_frame.spines['left'].set_visible(False)
        den_frame.spines['right'].set_visible(False)
        savepath = path + '/' + str(i) + '.jpg'
        plt.savefig(savepath, \
                    bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()


def randomcrop(img,randommax=None,randomaxis=None):

    tw, th = cfg_data.TRAIN_SIZE
    if randommax is not None:
        for j in range(len(randommax)):
            randommax[j][0]=randommax[j][0]-cfg_data.TRAIN_SIZE[0] if randommax[j][0]>cfg_data.TRAIN_SIZE[0] else 0#Xw
            randommax[j][1] = randommax[j][1] - cfg_data.TRAIN_SIZE[1] if randommax[j][1] > cfg_data.TRAIN_SIZE[1] else 0#Yh
        if randomaxis is None:
            randomaxis=torch.zeros((img.size()[0],2),dtype=int)
            for j in range(len(randommax)):
                randomaxis[j][0]=random.randint(0,randommax[j][0])#w
                randomaxis[j][1]=random.randint(0,randommax[j][1])#h
    if len(img.size())==3:
        size=(len(img),th,tw)
    else:
        size = (len(img),3, th, tw)
    output = torch.cuda.FloatTensor(size) if torch.cuda.is_available() else torch.FloatTensor(size)
    torch.zeros(size, out=output)

    if len(img.size())==3:
        for i in range(len(img)):
            output[i] = img[i,randomaxis[i][1]:randomaxis[i][1]+th,randomaxis[i][0]:randomaxis[i][0]+tw]
    else:
        for i in range(len(img)):
            output[i]=img[i,:,randomaxis[i][1]:randomaxis[i][1]+th,randomaxis[i][0]:randomaxis[i][0]+tw]

    return output,randomaxis

if __name__ == '__main__':
    import torch
    # Set manual seed for PyTorch
    if cfg.seed == 0:
        print('Using random seed.')
        torch.backends.cudnn.benchmark = True
    else:
        print('Using manual seed:', cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    pwd = os.path.split(os.path.realpath(__file__))[0]
    trainer = DATrainer(cfg,torch.cuda.device_count(),pwd)
    trainer.train()


