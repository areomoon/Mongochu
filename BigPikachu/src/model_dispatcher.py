import models

MODEL_DISPATCHER = {
    'resnet34': models.ResNet34(pretrained=True, n_class=3),
    'resnet34_eval': models.ResNet34(pretrained=False, n_class=3),
    'vgg16': models.VGG16(pretrained=True, n_class=3),
    'vgg16_eval': models.VGG16(pretrained=False, n_class=3),
    'se_resnext101_32x4d':models.SE_ResNext101_32x4d(pretrained=True, n_class=3),
    'se_resnext101_32x4d_eval':models.SE_ResNext101_32x4d(pretrained=False, n_class=3),
    'se_resnext101_32x4d_sSE':models.SE_ResNext101_32x4d_sSE(pretrained=True, n_class=3),
    'se_resnext101_32x4d_sSE_eval':models.SE_ResNext101_32x4d_sSE(pretrained=False, n_class=3)
}