
from efficientnet_pytorch import EfficientNet
import torch
from torch import nn


class DownstreamClassifer(nn.Module):
    def __init__(self, no_of_classes =256,final_pooling_type="Avg"):
        super(DownstreamClassifer, self).__init__()
        self.backbone = EfficientNet.from_name('efficientnet-b0',
                                                    final_pooling_type=final_pooling_type,
                                                    include_top = False,
                                                    in_channels = 1,
                                                    image_size = None)
        self.classifier = nn.Linear(1280,no_of_classes)

    def forward(self,batch):
        x = self.backbone(batch)
        x = x.flatten(start_dim=1) #1280 (already swished)
        x = self.classifier(x)        
        return x

