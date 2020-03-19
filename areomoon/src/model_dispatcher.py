import models

MODEL_DISPATCHER = {
    'resnet34': models.ResNet34(pretrained=True, n_class=3),
    'resnet34_eval': models.ResNet34(pretrained=False, n_class=3)
}