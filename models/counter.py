import  torch
import torch.nn as nn
import torch.nn.functional as F
from models.Vgg import VGGnet
from models.csrnet import CSRNet
from models.decoder import decoder
from collections import OrderedDict
from config import cfg
import torch.utils.model_zoo as model_zoo

class Counter(nn.Module):
    def __init__(self,pretrained=cfg.pre):
        super().__init__()
        self.encoder = VGGnet()
        self.decoder = decoder(feature_channel=512)
        if pretrained:
            self.encoder.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth'), strict=False)
        self.dropout=nn.Dropout(cfg.dropout)
    def forward(self, inp,noise=False):
        inp=self.encoder(inp)
        if noise:
            inp=self.dropout(inp)
        inp=self.decoder(inp)
        return inp

if __name__== '__main__':
    from  torchsummary import summary
    model = Counter(True).cuda()
    summary(model,(3, 768, 978), batch_size=1)


