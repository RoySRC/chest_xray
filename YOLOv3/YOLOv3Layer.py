from collections import OrderedDict

import torch.nn as nn


class YOLOv3Layer(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(YOLOv3Layer, self).__init__()
        self.num_anchors = 4

        self.conv1 = self.__conv2d__(in_channel, 512, 1)
        self.conv2 = self.__conv2d__(512, 1024, 3)
        self.conv3 = self.__conv2d__(1024, 512, 1)
        self.conv4 = self.__conv2d__(512, 1024, 3)
        self.conv5 = self.__conv2d__(1024, 512, 1)
        self.conv6 = self.__conv2d__(512, 1024, 3)
        self.out = nn.Conv2d(in_channels=1024,
                             out_channels=self.num_anchors * (4 + 1 + n_classes),
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)

    def __conv2d__(self, in_channel, out_channel, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=(kernel_size - 1) // 2 if kernel_size else 0,
                      bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, v):
        o = self.conv1(v)
        o = self.conv2(o)
        o = self.conv3(o)
        o = self.conv4(o)
        b = self.conv5(o)
        o = self.conv6(b)
        o = self.out(o)
        return o, b
