import torchvision
import torch.nn as nn

from torchvision.models.detection.ssd import (
    SSD, 
    DefaultBoxGenerator,
    SSDHead
)

import torchvision
import torch.nn as nn
from torchvision.models.detection.ssd import (
    SSD, 
    DefaultBoxGenerator,
    SSDHead
)
from collections import OrderedDict

class ResNetBackbone(nn.Module):
    def __init__(self, original_resnet):
        super().__init__()
        self.conv1 = original_resnet.conv1
        self.bn1 = original_resnet.bn1
        self.relu = original_resnet.relu
        self.maxpool = original_resnet.maxpool
        self.layer1 = original_resnet.layer1
        self.layer2 = original_resnet.layer2
        self.layer3 = original_resnet.layer3
        self.layer4 = original_resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)  # 256 channels
        f2 = self.layer2(f1) # 512 channels
        f3 = self.layer3(f2) # 1024 channels
        f4 = self.layer4(f3) # 2048 channels

        # return [f1, f2, f3, f4]
        return OrderedDict([
            ("0", f1),
            ("1", f2),
            ("2", f3),
            ("3", f4),
        ])

def create_model(num_classes=91, size=300, nms=0.45):

    resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    backbone = ResNetBackbone(resnet)
    out_channels = [256, 512, 1024, 2048]

    anchor_generator = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3]],
    )
    num_anchors = anchor_generator.num_anchors_per_location()
    head = SSDHead(out_channels, num_anchors, num_classes)
    model = SSD(
        backbone=backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        size=(size, size),
        head=head,
        nms_thresh=nms
    )

    for module in [backbone.conv1, backbone.bn1, backbone.layer1]:
        for param in module.parameters():
            param.requires_grad = False

    return model