
import torch.nn as nn
from torchvision import models
import pretrainedmodels

from .config import args

def resnet50(pretrained=True):
    """
    function to get pretrained model resnet50 
    --
    INPUTS:
    pretrained: (bool) 
    --
    OUTPUTS: model
    """ 
    model = models.resnet50(pretrained=pretrained)
    return model

def se_resnext50_32x4d(pretrained=True):
    """
    function to get pretrained model resnext50_32x4d
    --
    INPUTS:
    pretrained: (bool) 
    --
    OUTPUTS: model
    """ 
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.se_resnext50_32x4d(pretrained=pretrained)
    return model

def se_resnext101_32x4d(pretrained=True):
    """
    function to get pretrained model resnext101_32x4d
    --
    INPUTS:
    pretrained: (bool) 
    --
    OUTPUTS: model
    """ 
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.se_resnext101_32x4d(pretrained=pretrained)
    return model

class ResNet50(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet50, self).__init__()
        self.backbone = model

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, num_classes)
        self.conv_last = nn.Conv2d(512, num_classes, 1)
        

    def forward(self, x):
        
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)


        x = self.backbone.avgpool(x)
        x = self.dropout(x)  

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class SEResnext101(nn.Module): 
    def __init__(self, model, num_classes=1000): 
        super().__init__() 
        self.backbone = model 
        self.dropout = nn.Dropout(0.5) 
        self.fc = nn.Linear(2048, 5) 
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x): 
        x = self.backbone.layer0(x) 
        x = self.backbone.layer1(x) 
        x = self.backbone.layer2(x) 
        x = self.backbone.layer3(x) 
        x = self.backbone.layer4(x) 
        x = self.avg_pool(x)
        x = self.dropout(x)  
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x