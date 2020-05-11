import torch.nn as nn
import torchvision
import pretrainedmodels
from torch.nn import functional as F

class ResNet34(nn.Module):
    def __init__(self,pretrained, n_class):
        super(ResNet34, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained=None)
        self.l0 = nn.Linear(512, n_class)

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size,-1)
        output = self.l0(x)

        return output

class VGG16(nn.Module):
    def __init__(self, pretrained, n_class):
        super(VGG16, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["vgg16"](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__["vgg16"](pretrained=None)

        self.l0 = nn.Linear(512, n_class)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(64, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        x = self.model._features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size,-1)

        # output = self.l0(x)
        output = self.classifier(x)

        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class SE_ResNext101_32x4d(nn.Module):
    def __init__(self, pretrained, n_class):
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
        self.dense = nn.Linear(2048,128)
        self.dropout = nn.Dropout(0.3)

        self.l0 = nn.Linear(128, n_class)

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
        x = self.dense(x)
        x = F.relu(x)
        x = self.dropout(x)
        output = self.l0(x)

        return output

class pnasnet5large(nn.Module):
    def __init__(self, pretrained, n_class):
        super(pnasnet5large, self).__init__()
        if pretrained:
            self.model = pretrainedmodels.__dict__["pnasnet5large"](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__["pnasnet5large"](pretrained=None)
        
        self.l0 = nn.Linear(2048, n_class)
        
    def forward(self, x):
        batch_size, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        output = self.l0(x)

        return output
        

class densenet201(nn.Module):

    def __init__(self):
        super(densenet201, self).__init__()
        
        self.base_model = torchvision.models.densenet201(pretrained=True)
        self.base_model.classifier = nn.Identity()
        self.fc = nn.Sequential(
                    nn.Linear(1920, 1024, bias = True),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(1024, 512, bias = True),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(512, 104))
        
    def forward(self, inputs):
        x = self.base_model(inputs)
        return self.fc(x)
