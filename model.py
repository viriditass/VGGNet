import torch
import torch.nn as nn  

VGG_types = {
    # VGG11
    'VGG-A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # VGG11 LRN
    'VGG-A-LRN': [64, 'L', 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # VGG13
    'VGG-B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # VGG16
    'VGG-D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    # VGG19
    'VGG-E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

## VGGNet
class VGGNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, VGG_type=VGG_types['VGG-A']):
        super(VGGNet, self).__init__()

        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_type)

        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            # Convolution Layer
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]

                in_channels = x
            # MaxPool
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
            # Local Response Normalization
            elif x == 'L':
                layers += [nn.LocalResponseNorm(2)]
                
        return nn.Sequential(*layers)