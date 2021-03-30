import torch.nn as nn
from DenseBlock import DenseBlock
from TransitionLayer import TransitionLayer

class DenseNet(nn.Module):
    def __init__(self, nr_classes):
        super(DenseNet, self).__init__()

        self.lowconv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, padding=3, bias=False)
        self.relu = nn.ReLU()

        # 416 -> 208
        self.denseblock1, self.transitionLayer1 = self.__make_layer__(3, 64, 32, 128)

        # 208 -> 104
        self.denseblock2, self.transitionLayer2 = self.__make_layer__(3, 128, 32, 128)

        # 104 -> 52
        self.denseblock3, self.transitionLayer3 = self.__make_layer__(3, 128, 32, 128)

        # 52 -> 26
        self.denseblock4, self.transitionLayer4 = self.__make_layer__(3, 128, 32, 128)

        # 26 -> 13
        self.denseblock5, self.transitionLayer5 = self.__make_layer__(3, 128, 32, 128)

        self.bn = nn.BatchNorm2d(num_features=128)
        self.classifier = nn.Linear(128 * 12**2, nr_classes)


    def __make_layer__(self, num_dense_block_layers, in_channels, dense_out_channels, out_channels):
        '''
        Make a dense net layer comprising of a densenet block followed by a transition block
        :param num_dense_block_layers: Number of convolution layers comprising each dense block
        :param in_channels: Number of input channels to dense block
        :param dense_out_channels: Number of output channels of dense block
        :param out_channels: Number of output channels of transition layer
        :return: a tuple containing the dense and transition blocks
        '''
        dense = DenseBlock(num_dense_block_layers,
                           in_channels=in_channels,
                           out_channels=dense_out_channels)
        transition = TransitionLayer(in_channels=num_dense_block_layers*dense_out_channels,
                                     out_channels=out_channels)
        return dense, transition


    def forward(self, x):
        out = self.relu(self.lowconv(x))

        out = self.denseblock1(out)
        out = self.transitionLayer1(out)

        out = self.denseblock2(out)
        out = self.transitionLayer2(out)

        out = self.denseblock3(out)
        out = self.transitionLayer3(out)

        out = self.denseblock4(out)
        out = self.transitionLayer4(out)

        out = self.denseblock5(out)
        out = self.transitionLayer5(out)

        out = self.bn(out)
        out = out.view(-1, 128 * 12**2)
        out = self.classifier(out)
        return out

    def probe(self, x):
        '''
        This is exactly the same as the forward function without the classification head, and that it
        returns three values instead of one.
        :param x:
        :return:
        '''

        out = self.relu(self.lowconv(x))

        out = self.denseblock1(out)
        out = self.transitionLayer1(out)

        out = self.denseblock2(out)
        out = self.transitionLayer2(out)

        out = self.denseblock3(out)
        out = self.transitionLayer3(out)
        small = out

        out = self.denseblock4(out)
        out = self.transitionLayer4(out)
        medium = out

        out = self.denseblock5(out)
        out = self.transitionLayer5(out)
        large = out

        return large, medium, small