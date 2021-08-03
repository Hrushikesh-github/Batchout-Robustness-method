import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("/home/hrushikesh/robust/vgg19/batchout_many")
from batchout_many import Batchout_Many

class VGG(nn.Module):
    def __init__(self, n):
        super(VGG, self).__init__()
        self.n = n
        self.features1 = self._make_layers(3, 64)
        self.features2 = self._make_layers(64, 64)
        self.features3 = self._make_layers(64, 'M')
        self.features4 = self._make_layers(64, 128)
        self.features5 = self._make_layers(128, 128)
        self.features6 = self._make_layers(128, 'M')
        self.features7 = self._make_layers(128, 256)
        self.features8 = self._make_layers(256, 256)
        self.features9 = self._make_layers(256, 'M')
        self.features10 = self._make_layers(256, 512)
        self.features11 = self._make_layers(512, 512)
        self.features12 = self._make_layers(512, 'M')
        self.classifier = nn.Linear(512, 10)
        self.batchout1 = Batchout_Many(self.n)
        self._initialize_weights()


    def forward(self, x, y):
        x = self.features1(x) #0
        x = self.features2(x) #1
        x = self.features3(x) #2 R
        x = self.features4(x) #3 0
        x = self.features5(x) #4 1
        x = self.features6(x) #5 R
        x = self.features7(x) #6 2
        x = self.features8(x) #7 3
        x = self.features8(x) #8 4
        x = self.features8(x) #9 5
        x = self.features9(x) #10 R

        x = self.features10(x) #11 6

        if self.training:
            x, r = self.batchout1(x, y)

        x = self.features11(x) #12

        if self.training:
            x, r = self.batchout1(x, y)

        x = self.features11(x) #13

        if self.training:
            x, _ = self.batchout1(x, y, r)

        x = self.features11(x) #14

        if self.training:
            x, r = self.batchout1(x, y)

        x = self.features12(x) #15 R

        if self.training:
            x, _ = self.batchout1(x, y, r)

        x = self.features11(x) #16

        if self.training:
            x, r = self.batchout1(x, y)

        x = self.features11(x) #17

        if self.training:
            x, _ = self.batchout1(x, y, r)

        x = self.features11(x) #18

        if self.training:
            x, r = self.batchout1(x, y)

        x = self.features11(x) #19

        if self.training:
            x, _ = self.batchout1(x, y, r)

        x = self.features12(x) #20 R

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _make_layers(self, in_channels, cfg):
        layers = []
        if cfg == 'M':
            # Apply max pooling to reduce dimension
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # Apply conv, there will be no reduction in dimensions
            layers += [nn.Conv2d(in_channels, cfg, kernel_size=3, padding=1),
                       nn.ReLU(inplace=True),
                       nn.BatchNorm2d(cfg)]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


