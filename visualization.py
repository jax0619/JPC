from matplotlib import pyplot as plt
import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd
from misc.utils import *
import scipy.io as sio
from PIL import Image, ImageOps
from misc.quality import get_psnr,get_ssim
from models.counter import NLT_Counter
from config import cfg
import tqdm

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

exp_name = './visual-display/'
dataset  = 'GCC2MALL'
da=True
type= 'da' if da else 'no_adapt'
den_path = os.path.join(exp_name+dataset,type)

print(den_path)

os.makedirs(den_path, mode=0o777, exist_ok=True)


class den_test:
    def __init__(self,model_path, tar_list,tarRoot,cfg_data,img_transform):
        self.cfg_data = cfg_data
        self.img_transform = img_transform
        self.tarRoot = tarRoot
        self.net =NLT_Counter()
        self.net.load_state_dict(torch.load(model_path))
        self.net = torch.nn.DataParallel(self.net).cuda()
        self.net.eval()
        with open(tar_list) as f:
            lines = f.readlines()
        self.tar_list = []
        for line in lines:
            line = line.strip('\n')
            self.tar_list.append(line)

    def forward(self):
        score = {'MAE':0, 'MSE':0, 'PSNR':0, 'SSIM':0}
        count = 0
        tar_list =  tqdm.tqdm(self.tar_list)
        best = []
        if not os.path.exists(den_path +'/'+ 'best'):
            os.mkdir(den_path +'/'+ 'best')
        for fname in tar_list:
            count +=1
            if dataset == 'GCC2SHHB' or dataset == 'GCC2MALL':
                add='.jpg'
            else:
                add=''
            imgname = os.path.join(self.tarRoot + "/test/img/" + fname +add)
            # imgname = os.path.join(self.tarRoot + "/test/img/" + fname )
            # filename_no_ext = filename.split('.')[0]
            denname = imgname.replace('img','den').replace('jpg','csv')
            # denname   = os.path.join(self.tarRoot + "/test/den/" + fname + ".csv")
            den = pd.read_csv(denname, sep=',',header=None).values
            den = den.astype(np.float32, copy=False)
            img = Image.open(imgname)

            if img.mode == 'L':
                img = img.convert('RGB')
            img = self.img_transform(img)
            gt = np.sum(den)
            img = img[None,:,:,:].cuda()

            pred_map = self.net(img)

            pred_map = pred_map.cpu().data.numpy()[0,0,:,:]
            pred = np.sum(pred_map)/self.cfg_data.LOG_PARA

            if abs(pred-gt)<max:
                best.append(fname.split('.')[0])

            score['MAE'] += np.abs(gt - pred)

            score['MSE'] += (gt - pred)*(gt - pred)
            score['SSIM']+= get_ssim(den, pred_map)
            score['PSNR']+= get_psnr(den, pred_map)

            pred_map = pred_map/np.max(pred_map+1e-20)

            den = den/np.max(den+1e-20)


            den_frame = plt.gca()
            plt.imshow(den,cmap='jet')
            den_frame.axes.get_yaxis().set_visible(False)
            den_frame.axes.get_xaxis().set_visible(False)
            den_frame.spines['top'].set_visible(False)
            den_frame.spines['bottom'].set_visible(False)
            den_frame.spines['left'].set_visible(False)
            den_frame.spines['right'].set_visible(False)
            plt.savefig(den_path+'/'+fname+'_gt_'+str(int(gt))+'.png',\
                bbox_inches='tight',pad_inches=0,dpi=600)
            if da:
                if abs(pred - gt) < max:
                    plt.savefig(den_path + '/' + 'best' + '/' + fname + '_gt_' + str(int(gt)) + '.png', \
                                bbox_inches='tight', pad_inches=0, dpi=600)
            else:
                if abs(pred - gt) > max:
                    plt.savefig(den_path + '/' + 'best' + '/' + fname + '_gt_' + str(int(gt)) + '.png', \
                                bbox_inches='tight', pad_inches=0, dpi=600)


            plt.close()

          # sio.savemat(exp_name+'/'+filename_no_ext+'_gt_'+str(int(gt))+'.mat',{'data':den})

            pred_frame = plt.gca()
            plt.imshow(pred_map,cmap='jet')
            pred_frame.axes.get_yaxis().set_visible(False)
            pred_frame.axes.get_xaxis().set_visible(False)
            pred_frame.spines['top'].set_visible(False)
            pred_frame.spines['bottom'].set_visible(False)
            pred_frame.spines['left'].set_visible(False)
            pred_frame.spines['right'].set_visible(False)
            plt.savefig(den_path+'/'+fname+'_'+type+'_'+str(float(pred))+'.png',\
                bbox_inches='tight',pad_inches=0,dpi=600)
            if da:
                if abs(pred - gt) < max:
                    plt.savefig(den_path + '/'+ 'best'+'/' + fname + '_DA_' + str(float(pred)) + '.png', \
                            bbox_inches='tight', pad_inches=0, dpi=600)
            else:
                if abs(pred - gt) > max:
                    plt.savefig(den_path + '/' + 'best' + '/' + fname + '_NoDA_' + str(float(pred)) + '.png', \
                                bbox_inches='tight', pad_inches=0, dpi=600)
            plt.close()


        score['MAE'], score['MSE'] = score['MAE'] / count, np.sqrt(score['MSE'] / count)
        score['SSIM'], score['PSNR'] = score['SSIM'] / count, score['PSNR'] / count
        print(best)
        print("processed   MAE_in: %.2f  MSE_in: %.2f"  % ( score['MAE'], score['MSE']))
        print("processed   PSNR: %.2f  SSIM: %.2f" %  ( score['PSNR'], score['SSIM']))
        # sio.savemat(exp_name+'/'+filename_no_ext+'_diff.mat',{'data':diff})

def get_pts(data):
    pts = []
    cols,rows = data.shape
    data = data*100

    for i in range(0,rows):  
        for j in range(0,cols):  
            loc = [i,j]
            for i_pt in range(0,int(data[i][j])):
                pts.append(loc) 
    return pts               


# label_path = '../ProcessedData/performed_bak_lite/scene_09_2/csv_den_maps_k15_s4_544_960/1531113494.csv'
label_path = '../ProcessedData/SHHB/train/den_10.28/99.csv'
def generater_hotmap(label_path):
    den = pd.read_csv(label_path, sep=',', header=None).values
    den = den.astype(np.float32, copy=False)
    gt = np.sum(den)
    den = den / np.max(den + 1e-20)
    den_frame = plt.gca()
    plt.imshow(den, cmap='jet')
    den_frame.axes.get_yaxis().set_visible(False)
    den_frame.axes.get_xaxis().set_visible(False)
    den_frame.spines['top'].set_visible(False)
    den_frame.spines['bottom'].set_visible(False)
    den_frame.spines['left'].set_visible(False)
    den_frame.spines['right'].set_visible(False)
    plt.savefig(den_path.replace('den_test','datu') + '/' + '99_fakel_label' + '_gt_' + str(int(gt)) + '.png', \
                bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close()



if __name__ == '__main__':

    from dataloader.setting import cfg_data
    mean_std = cfg_data.MEAN_STD

    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    restore = standard_transforms.Compose([
        standard_transforms.ToPILImage()
    ])
    pil_to_tensor = standard_transforms.ToTensor()
    if dataset=='GCC2SHHA':
        tarRoot = cfg_data.SHHA_DATA_PATH
        tar_test_list = os.path.join( cfg_data.SHHA_scene_dir+'/test', 'test.txt')
        cc_path = './exp/SHHA/da/best_few_shot/mae_86.7_mse_132.4/all_ep_261_mae_86.7_mse_132.4_th.pth' if da else \
            './exp/pre/04-27_23-16_SHHB_vgg16__lr1e-05_pre_train/all_ep_50_mae_27.3_mse_68.5.pth'
        max=43 if da else 188
    if dataset=='GCC2SHHB':
        tarRoot = cfg_data.SHHB_DATA_PATH
        tar_test_list = os.path.join(cfg_data.SHHB_scene_dir+'/test', 'test.txt')
        cc_path = './exp/SHHB/da/best_few_shot/mae_10.5_mse_15.6/all_ep_49_mae_10.5_mse_15.6_stu.pth' if da else\
            './exp/pre/04-27_23-16_SHHB_vgg16__lr1e-05_pre_train/all_ep_50_mae_27.3_mse_68.5.pth'
        max = 5 if da else 32
    if dataset=='GCC2QNRF':
        tarRoot = cfg_data.QNRF_DATA_PATH
        tar_test_list = os.path.join(cfg_data.QNRF_scene_dir+'/test', 'test.txt')
        cc_path = './exp/QNRF/da/best_few_shot/06-29_20-49_QNRF_vgg16__lr1e-05_train_train_few_shot/all_ep_199_mae_149.6_mse_249.9_th.pth' if da  else \
            './exp/pre/04-27_23-16_SHHB_vgg16__lr1e-05_pre_train/all_ep_50_mae_27.3_mse_68.5.pth'
        max = 75 if da else 276

    if dataset=='GCC2MALL':
        tarRoot = cfg_data.MALL_DATA_PATH
        tar_test_list = os.path.join(cfg_data.MALL_scene_dir+'/test', 'test.txt')
        cc_path = './exp/MALL/da/best_few_shot/mae 1.96 mse 2.53 08-30_11-08_MALL_vgg16__lr1e-05_train_train_few_shot/all_ep_99_mae_2.0_mse_2.5_stu.pth' if da else \
            './exp/pre/04-27_23-16_SHHB_vgg16__lr1e-05_pre_train/all_ep_50_mae_27.3_mse_68.5.pth'
        max = 1 if da else 6.2
    if dataset == 'GCC2WE':
        tarRoot = cfg_data.WE_DATA_PATH
        tar_test_list = os.path.join(cfg_data.WE_scene_dir + '/test', '200608.txt')
        cc_path = './exp/WE/03-01_01-55_WE_vgg16__lr1e-05_gamma0.98_IFS/all_ep_296_mae_4.4_mse_0.0.pth'
    if dataset=='GCC2UCSD':
        tarRoot = cfg_data.UCSD_DATA_PATH
        tar_test_list = os.path.join(cfg_data.UCSD_scene_dir+'/test', 'test.txt')
        cc_path = './exp/UCSD/02-29_00-42_UCSD_vgg16__lr1e-05_gamma0.98_NLT/all_ep_56_mae_1.5_mse_2.0.pth'

    den_test = den_test(cc_path,tar_test_list,tarRoot,cfg_data,img_transform)
    den_test.forward()
    # generater_hotmap(label_path)