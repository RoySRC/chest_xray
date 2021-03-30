import torch
import torch.nn as nn
from YOLOv3Layer import YOLOv3Layer

class YOLOv3Head(nn.Module):
    '''
    YOLO head comprised of multiple YOLO layers, one for each object scale.
    '''
    def __init__(self, in_channels, n_classes):
        '''

        :param in_channels: array containing channels for large, medium, small pbject
        :param n_classes:
        '''
        super(YOLOv3Head, self).__init__()

        self.n_classes = n_classes
        self.large_scale_layer = YOLOv3Layer(in_channels[0], n_classes)
        self.medium_scale_layer = YOLOv3Layer(in_channels[1], n_classes)
        self.small_scale_layer = YOLOv3Layer(in_channels[2], n_classes)

        self.medium_scale_upsample = nn.Upsample(scale_factor=25/12,
                                                 mode='nearest')
        self.small_scale_upsample = nn.Upsample(scale_factor=51/25,
                                                mode='nearest')


    def forward(self, v1, v2, v3):
        '''

        :param v1: feature vector containing information about large scale objects
        :param v2: feature vector containing information about medium scale objects
        :param v3: feature vector containing information about small scale objects
        :return:
        '''
        large_output, large_branch = self.large_scale_layer(v1)

        medium_upsample = self.medium_scale_upsample(large_branch)
        cat_v2_medium_upsample = torch.cat([v2, medium_upsample], 1)
        medium_output, medium_branch = self.medium_scale_layer(cat_v2_medium_upsample)

        small_upsample = self.small_scale_upsample(medium_branch)
        cat_v3_small_upsample = torch.cat([v3, small_upsample], 1)
        small_output, _ = self.small_scale_layer(cat_v3_small_upsample)

        return large_output, medium_output, small_output