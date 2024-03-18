# %%
import torch.nn as nn
import torch
from torchvision import models
import collections
import torch.utils.model_zoo as model_zoo
from models.decoder import decoder
from config import cfg
from models.Mixstyle import MixStyle
class CSRNet(nn.Module):
    def __init__(self,pretrained=cfg.pre,bn=False):
        super(CSRNet, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128,
                              'M', 256, 256, 256, 'M', 512, 512, 512]
        # self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.backend_feat = [512, 512, 512]
        self.mix=cfg.feature
        if self.mix:
            self.mixstyle = MixStyle(p=0.5, alpha=0.1)
            self.mixstyle.set_activation_status(self.mix)

        self.frontend = self.make_layers(self.frontend_feat,batch_norm=bn)
        self.backend = self.make_layers(
            self.backend_feat, in_channels=512, dilation=True,batch_norm=bn)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        # if not load_weights:
        #     self.frontend.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth'), strict=False)
        #     print('initial successfully')
        if pretrained:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            fsd = collections.OrderedDict()
            # 10 convlution *(weight, bias) = 20 parameters
            for i in range(len(self.frontend.state_dict().items())):
                temp_key = list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)
        self.dropout=nn.Dropout(cfg.dropout)
        self.decoder = decoder(feature_channel=512)
    def forward(self, x,noise=False):
        x = self.frontend(x)
        if noise:
            x=self.dropout(x)
        x = self.backend(x)
        x= self.decoder(x)
        # x = self.output_layer(x)
        # x = nn.functional.interpolate(x, scale_factor=8)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layers(self,config, in_channels=3, batch_norm=False, dilation=False):
        if dilation:
            d_rate = 2
        else:
            d_rate = 1
        layers = []
        for v in config:
            if v == 'M':
                if self.mix:
                    layers += [self.mixstyle]
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                                   padding=d_rate, dilation=d_rate)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


# testing code
if __name__ == "__main__":
    csrnet = CSRNet().cuda()

    from torchsummary import summary


    # para.requires_grad = False
    # for i,b in model.state_dict().items():
    #     print(i,b.shape)
    summary(csrnet, (3, 768, 978), batch_size=1)

