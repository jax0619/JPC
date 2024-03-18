import torch
import torch.nn as nn
from config import cfg
from models.Mixstyle import MixStyle
VGG_types = {
"VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
"VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
"VGG16": [64,64,"M",128,128,"M",256,256,256,"M",512,512,512],
#"VGG16": [64,64,"M",128,128,"M",256,256,256,"M",512,512,512,"M",512,512,512,"M",],
"VGG19": [64,64,"M",128,128,"M",256,256,256,256,"M",512,512,512,512,
          "M",512,512,512,512,"M",],}


VGGType = "VGG16"
class VGGnet(nn.Module):
    def __init__(self, in_channels=3):
        super(VGGnet, self).__init__()
        self.in_channels = in_channels
        self.mix=cfg.feature
        if self.mix:
            self.mixstyle = MixStyle(p=0.5, alpha=0.1)
            self.mixstyle.set_activation_status(self.mix)
        self.features = self.create_conv_layers(VGG_types[VGGType])
        self.conv1=nn.Conv2d(3,3,3)
        self.linear=nn.Linear(18,2)
    def forward(self, x):
        # x = self.mixstyle(x)
        x = self.features(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [
                nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                )
                ]
                layers += [
                    nn.ReLU()
                ]
                in_channels = x
            elif x == "M":
                if self.mix:
                    layers += [self.mixstyle]
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        return nn.Sequential(*layers)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGnet(in_channels=3).to(device)
    print(model)
    from torchsummary import summary

    summary(model, (3, 224, 224), batch_size=1)
    x = torch.randn(1, 3, 224, 224).to(device)
    print(model(x).shape)
