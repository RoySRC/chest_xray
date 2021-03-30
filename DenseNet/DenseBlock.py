import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, layers, in_channels, out_channels):
        '''
        DenseBlock. The height and width of the output feature maps is the same as the height and width of
        the input feature maps.
        :param name: The name of the DenseBlock
        :param layers: The number of convolution layers in this block
        :param in_channels: The number of input channels/feature maps/filters
        :param out_channels: The number of output channels/feature maps/filters
        '''
        super(DenseBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(num_features=in_channels)

        self.layers = []
        _in = in_channels
        for l in range(layers):
            self.layers.append(
                nn.Conv2d(in_channels = _in,
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1)
            )
            self.add_module(f'conv_{l}', self.layers[-1])
            _in = (l+1)*out_channels

    def forward(self, X):
        Y = self.bn(X)
        concat = []
        for i, layer in enumerate(self.layers):
            Y = self.relu(layer(Y))
            concat.append(Y)
            if i == 0:
                continue
            dense = self.relu(torch.cat(concat, 1))
            Y = dense

        return Y