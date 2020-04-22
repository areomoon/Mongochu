import torch.nn as nn
import pretrainedmodels
from efficientnet_pytorch import EfficientNet
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

class sSE_Block(nn.Module):
    """
       Re-implementation of Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks
       (Abhijit Guha Roy, et al., MICCAI 2018)
    """
    def __init__(self):
        super(sSE_Block, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=(1,1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv2d(x)
        x1 = self.sigmoid(x1)
        return x * x1

class SE_ResNext101_32x4d_sSE(nn.Module):
    def __init__(self,pretrained, n_class):
        super(SE_ResNext101_32x4d_sSE, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["se_resnext101_32x4d"](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__["se_resnext101_32x4d"](pretrained=None)

        self.sSE_Block = sSE_Block()
        self.dropout = nn.Dropout(0.2)
        self.l0 = nn.Linear(2048,n_class)

    def forward(self, x):
        '''
        WIP
        :param x:
        :return:
        '''
        batch_size, _, _, _ = x.shape
        x = self.model.features(x)
        x = self.sSE_Block(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size,-1)
        x = F.relu(x)
        x = self.dropout(x)
        output = self.l0(x)

        return output


class  EfficientNet_B6(nn.Module):
    def __init__(self,pretrained, n_class):
        super(EfficientNet_B6, self).__init__()
        if pretrained is True:
            self.model = EfficientNet.from_pretrained('efficientnet-b6')

        self.l0 = nn.Linear(2304,n_class)

    def forward(self, x):
        '''
        WIP
        :param x:
        :return:
        '''
        batch_size, _, _, _ = x.shape
        x = self.model.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size,-1)
        output = self.l0(x)

        return output





