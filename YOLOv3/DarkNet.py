import math
from collections import OrderedDict

from torch import optim

from model import model_base
import torch, time
import torch.nn as nn

'''
* Residual structure
* Use a 1x1 convolution to decrease the number of channels, then use a 3x3 convolution to extract features 
  and increase the number of channels.
* Finally, connect a residual edge
'''
class ResidualBlock(nn.Module, model_base):
    def __init__(self, inplanes, planes):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


class DarkNet_53(nn.Module, model_base):
    def __init__(self, layers, image_dims, learning_rate=0.01):
        super(DarkNet_53, self).__init__()
        self.inplanes = 32
        self.img_h, self.img_w = image_dims
        # 416,416,3 -> 416,416,32
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)

        self.resnet_blocks = []
        for i,l in enumerate(layers):
            self.resnet_blocks.append(self._make_layer([self.inplanes, 2*self.inplanes], l))
            self.add_module(f'layer{i}', self.resnet_blocks[-1])

        self.avg_pool = nn.AvgPool2d(kernel_size=(13, 13), stride=1)
        self.flatten = nn.Flatten()
        self.connected = nn.Linear(in_features=1024, out_features=15)
        # self.sigmoid = nn.Sigmoid()
        # self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        # self.loss = nn.BCELoss()
        self.layers_out_filters = [64, 128, 256, 512, 1024]

        # Perform weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, planes, blocks):
        '''
        In each layer, first use a 3x3 convolution with a step size of 2 for downsampling
        Then stack the residual structure
        '''
        layers = []
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3,
                                            stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))

        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append((f"residual_{i}", ResidualBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        last3 = []
        for i,resnet_block in enumerate(self.resnet_blocks):
            x = resnet_block(x)
            if i > len(self.resnet_blocks) - 4:
                last3.append(x)
        out3, out4, out5 = last3

        # x = self.layer1(x)
        # x = self.layer2(x)
        # out3 = self.layer3(x)
        # out4 = self.layer4(out3)
        # out5 = self.layer5(out4)

        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.connected(x)
        return x

    # def backward(self, Y_hat, targets):
    #     self.optimizer.zero_grad()
    #     loss = self.loss(Y_hat, targets)
    #     loss_str = loss.item()
    #     loss.backward()
    #     self.optimizer.step()
    #     return loss_str

def darknet53(pretrained, **kwargs):
    model = DarkNet_53([1, 2, 8, 8, 4])
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model

if __name__ == "__main__":
    image_dims = (3408, 3320)
    batch_size = 2
    darknet_model = DarkNet_53([1,2,8,8,4], image_dims).cuda()
    print(darknet_model)
    for i in range(10):
       y_hat = darknet_model(torch.ones((2, 1, 1024, 1024)).cuda())
       target = torch.zeros((2, 15)).cuda()
       print(darknet_model.backward(y_hat, target))
    print("sleeping")
    time.sleep(15)
