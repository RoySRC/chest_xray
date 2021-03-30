import torch
import torch.nn as nn

import sys
sys.path.insert(1, '../DenseNet')
from DenseNet import DenseNet

from YOLOv3Head import YOLOv3Head

class YOLOv3(nn.Module):
    def __init__(self, in_channels, n_classes, backbone_modelfile_path):
        '''

        :param in_channels:
        :param n_classes: the number of classes. This also contains the empty class
        :param backbone_modelfile_path:
        '''
        super(YOLOv3, self).__init__()

        # Load the backbone
        self.backbone = DenseNet(n_classes).eval()
        checkpoint = torch.load(backbone_modelfile_path)
        self.backbone.load_state_dict(checkpoint['state_dict'])

        self.yolo_head = YOLOv3Head(in_channels, n_classes-1)

    def eval(self):
        self.backbone = self.backbone.eval()
        self.yolo_head = self.yolo_head.eval()

    def train(self):
        self.backbone = self.backbone.eval()
        self.yolo_head = self.yolo_head.train()

    def forward(self, X):
        large, medium, small = self.backbone.probe(X)
        large, medium, small = self.yolo_head(large, medium, small)
        return large, medium, small