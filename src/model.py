import torch
from torch import nn
from torchvision import models


class NetClient1(nn.Module):
    def __init__(self, in_, num_class):
        super(NetClient1, self).__init__()
        resnet50 = models.resnet50(pretrained=False, num_classes=num_class)
        self.resnet_post = nn.Sequential(nn.Conv2d(in_, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                         resnet50.bn1,
                                         resnet50.relu,
                                         resnet50.maxpool)

    def forward(self, x):
        x = self.resnet_post(x)
        return x


class NetServer(nn.Module):
    def __init__(self, num_class):
        super(NetServer, self).__init__()
        resnet50 = models.resnet50(pretrained=False, num_classes=num_class)
        self.resnet_medium = nn.Sequential(resnet50.layer1,
                                           resnet50.layer2,
                                           resnet50.layer3,
                                           resnet50.layer4)

    def forward(self, x):
        x = self.resnet_medium(x)
        return x


class NetClient2(nn.Module):
    def __init__(self, num_class):
        super(NetClient2, self).__init__()
        resnet50 = models.resnet50(pretrained=False, num_classes=num_class)
        self.avgpool = resnet50.avgpool
        self.fc = resnet50.fc

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class NetSurrogate(nn.Module):
    def __init__(self, in_, num_class):
        super(NetSurrogate, self).__init__()
        resnet50 = models.resnet50(pretrained=False, num_classes=num_class)
        self.net_former = nn.Sequential(nn.Conv2d(in_, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False), )
        self.resnet = nn.Sequential(resnet50.layer1,
                                    resnet50.layer2,
                                    resnet50.layer3,
                                    resnet50.layer4,
                                    )
        self.net_surrogate = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=1372, kernel_size=1),
                                           nn.Conv2d(in_channels=1372, out_channels=10, kernel_size=1), )

    def forward(self, x):
        x = self.net_former(x)
        x = self.resnet(x)
        x = self.net_surrogate(x)
        x = torch.squeeze(x)
        return x