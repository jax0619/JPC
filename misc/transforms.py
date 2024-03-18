import numbers
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from config import cfg
import torch
import torchvision.transforms as tf
import cv2
import torch.nn as nn


# ===============================img tranforms============================

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, bbx=None):
        if bbx is None:
            for t in self.transforms:
                if t is None:
                    continue
                img, mask = t(img, mask)
            return img, mask
        for t in self.transforms:
            img, mask, bbx = t(img, mask, bbx)
        return img, mask, bbx

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, bbx=None,y=0):
        if random.random() < 0.5 :# 随机生成0-1之间的浮点数 ，每次执行生成的不一样

            if bbx is None:
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            w, h = img.size
            xmin = w - bbx[:, 3]
            xmax = w - bbx[:, 1]
            bbx[:, 1] = xmin
            bbx[:, 3] = xmax
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), bbx
        else:
            return img,mask
        # if bbx is None:
        #     return img, mask
        #return img, mask, bbx
class Randomresize(object):
    def __init__(self):
        self.ratio=random.randrange(707,1414)
        self.ratio=self.ratio/1000.0
    def __call__(self, img, mask):
        if random.random() < 0.5:
            w,h=img.size
            new_w=int(w*self.ratio/8)*8
            new_h=int(h*self.ratio/8)*8
            img=img.resize((new_w,new_h),Image.BILINEAR)
            mask=mask.resize((new_w,new_h),Image.NEAREST)
        return img,mask


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding


    def __call__(self, img, mask,x1=None,y1=None,randommax=None):
        # if self.padding > 0:
        #     img = ImageOps.expand(img, border=self.padding, fill=0)
        #     mask = ImageOps.expand(mask, border=self.padding, fill=0)
        #
        # assert img.size == mask.size
        w, h = img.size
        tw, th  = self.size
        if w == tw and h == th:
            return img, mask,w,h
        hpad = th - h
        wpad = tw - w

        if hpad > 0:
            img = ImageOps.expand(img, (0, 0, 0, hpad))
            mask = ImageOps.expand(mask, (0, 0, 0, hpad))
            randommaxh=0
        else:
            randommaxh=h-th

        if wpad > 0:
            img = ImageOps.expand(img, (0, 0, wpad, 0))
            mask = ImageOps.expand(mask, (0, 0, wpad, 0))
            randommaxw = 0
        else:
            randommaxw = w - th

        if x1 == None and y1 == None:
            x1 = random.randint(0, randommaxw)
            y1 = random.randint(0, randommaxh)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)), x1, y1

class RandGaussianBlur(object):

    # Note: there is significant difference
    # with OpenCV implementation how sigma is
    # computed for the given radius

    def __init__(self, radius=cfg.RandGaussianBlur):
        self.radius = radius

    def __call__(self, img, mask,ns):

        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(self.radius))
            ns[0]=1
        return img, mask,ns

class MaskRandJitter(object):

    # Note: there is significant difference
    # with OpenCV implementation how sigma is
    # computed for the given radius

    def __init__(self, p=0.5):
        group=[[1,1,1,0],[1,0,1,1]]
        numbers=group[random.randint(0,1)]
        value=[a*b for a,b in zip(numbers,cfg.MaskRandJitter)]
        # value=cfg.MaskRandJitter
        self.jitter = tf.ColorJitter(brightness=value[0], \
                                     contrast=value[1], \
                                     saturation=value[2], \
                                     hue=value[3])
        # self.jitter = tf.ColorJitter(brightness=cfg.MaskRandJitter[0], \
        #                              contrast=cfg.MaskRandJitter[1], \
        #                              saturation=cfg.MaskRandJitter[2], \
        #                              hue=cfg.MaskRandJitter[3])
    def __call__(self, img, mask,ns):

        if random.random() < 0.5:
            img = self.jitter(img)
            ns[1]=1
        return img, mask,ns

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))



class FreeScale(object):
    def __init__(self, size):
        self.size = size  # (h, w)

    def __call__(self, img, mask):
        return img.resize((self.size[1], self.size[0]), Image.BILINEAR), mask.resize((self.size[1], self.size[0]), Image.NEAREST)


class ScaleDown(object):
    def __init__(self, size):
        self.size = size  # (h, w)

    def __call__(self, mask):
        return  mask.resize((self.size[1]/cfg.TRAIN.DOWNRATE, self.size[0]/cfg.TRAIN.DOWNRATE), Image.NEAREST)


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        if img.size != mask.size:
            print (img.size)
            print (mask.size)           
        assert img.size == mask.size
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img, mask
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)

class Batch_Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (N,C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        # if not _is_tensor_image(tensor):
        #     raise TypeError('tensor is not a torch image.')
        #
        # if not inplace:
        #     tensor = tensor.clone()


        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=tensor.device)
        std =  torch.as_tensor(self.std,  dtype=torch.float32, device=tensor.device)

        new_tensor = torch.clone(tensor)   #don't change the input tensor
        for i in range(new_tensor.size()[0]):
            new_tensor[i].sub_(mean[:, None, None]).div_(std[:, None, None])

        return new_tensor


# ===============================label tranforms============================
class Batch_DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):


        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=torch.float32, device=tensor.device)
        new_tensor = torch.clone(tensor)  # don't change the input tensor
        for i in range(new_tensor.size()[0]):
            new_tensor[i].mul_(std[:, None, None]).add_(mean[:, None, None])

        # print(torch.max(new_tensor),torch.min(new_tensor))
        return new_tensor

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        print(self.mean)
        print(self.std)
    def __call__(self, tensor):
        new_tensor = torch.clone(tensor)  # don't change the input tensor
        for t, m, s in zip(new_tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        # print(torch.max(new_tensor), torch.min(new_tensor))
        return new_tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class LabelNormalize(object):
    def __init__(self, para):
        self.para = para

    def __call__(self, tensor):
        # tensor = 1./(tensor+self.para).log()
        tensor = torch.from_numpy(np.array(tensor))
        tensor = tensor*self.para
        return tensor

class GTScaleDown(object):
    def __init__(self, factor=8):
        self.factor = factor

    def __call__(self, img):
        w, h = img.size
        if self.factor==1:
            return img
        tmp = np.array(img.resize((w/self.factor, h/self.factor), Image.BICUBIC))*self.factor*self.factor
        img = Image.fromarray(tmp)
        return img