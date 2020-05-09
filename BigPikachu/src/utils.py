def model_dispatcher(if_pretrain, base_model, nclass):
    if base_model == 'se_resnext101_32x4d':
        return models.SE_ResNext101_32x4d(pretrained=if_pretrain, n_class=nclass)

    elif base_model == 'vgg16':
        return models.VGG16(pretrained=if_pretrain, n_class=nclass)

    elif base_model == 'resnet34': 
        return models.ResNet34(pretrained=if_pretrain, n_class=nclass)
    
    elif base_model == 'se_resnext101_32x4d_sSE': 
        return models.se_resnext101_32x4d_sSE(pretrained=if_pretrain, n_class=nclass)