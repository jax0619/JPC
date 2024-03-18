import torch.nn as nn
import torch
import torch.nn.functional as F
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, stride=1,padding=0 ,noise=False,use_bn=False):
        super(BasicConv, self).__init__()
        self.ues_bn = use_bn
        self.noise = noise

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding)
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        x = self.conv(x)
        if self.ues_bn:
            x = self.bn(x)

        return F.relu(x, inplace=True)

class decoder(nn.Module):
    def __init__(self,feature_channel=512):
        super(decoder,self).__init__()
        self.de_pred = nn.Sequential(
            BasicConv(feature_channel, 256,kernel_size=1,use_bn=True),
            nn.UpsamplingNearest2d(scale_factor=2),

            BasicConv(256, 128, kernel_size=3, padding=1, use_bn=True),
            nn.UpsamplingNearest2d(scale_factor=2),

            BasicConv(128, 64, kernel_size=3, padding=1, use_bn=True),
            nn.UpsamplingNearest2d(scale_factor=2),

            BasicConv(64, 64, kernel_size=3, padding=1, use_bn=True),
            BasicConv(64, 1, kernel_size=1,  padding=0, use_bn=False)
        )
    def forward(self, x):
        x = self.de_pred(x)
        return x


if __name__== '__main__':
    from  torchsummary import summary
    model = decoder().cuda()
    summary(model,(512,96,122), batch_size=5)