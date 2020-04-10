import torch.nn as nn
import pretrainedmodels
from torch.nn import functional as F

class ResNet34(nn.Module):
    def __init__(self,pretrained, n_class):
        super(ResNet34, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained=None)
        self.l0 = nn.Linear(512,n_class)

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size,-1)
        output = self.l0(x)

        return output

class VGG16(nn.Module):
    def __init__(self,pretrained, n_class):
        super(VGG16, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["vgg16"](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__["vgg16"](pretrained=None)

        self.l0 = nn.Linear(512,n_class)

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        x = self.model._features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size,-1)
        output = self.l0(x)

        return output

class SE_ResNext101_32x4d(nn.Module):
    def __init__(self,pretrained, n_class):
        super(SE_ResNext101_32x4d, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["se_resnext101_32x4d"](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__["se_resnext101_32x4d"](pretrained=None)

        self.l0 = nn.Linear(2048,n_class)

    def forward(self, x):
        '''
        WIP
        :param x:
        :return:
        '''
        batch_size, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size,-1)
        output = self.l0(x)

        return output

