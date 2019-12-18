import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# code from https://github.com/irasin/Pytorch_WCT

class Interpolate(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor)
        return x


vgg_decoder_relu5_1 = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    Interpolate(2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, 3),
    nn.ReLU(),
    Interpolate(2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, 3),
    nn.ReLU(),
    Interpolate(2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, 3),
    nn.ReLU(),
    Interpolate(2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, 3)
    )


class Decoder(nn.Module):
    def __init__(self, level, pretrained_path=None):
        super().__init__()
        if level == 1:
            self.net = nn.Sequential(*copy.deepcopy(list(vgg_decoder_relu5_1.children())[-2:]))
        elif level == 2:
            self.net = nn.Sequential(*copy.deepcopy(list(vgg_decoder_relu5_1.children())[-9:]))
        elif level == 3:
            self.net = nn.Sequential(*copy.deepcopy(list(vgg_decoder_relu5_1.children())[-16:]))
        elif level == 4:
            self.net = nn.Sequential(*copy.deepcopy(list(vgg_decoder_relu5_1.children())[-29:]))
        elif level == 5:
            self.net = nn.Sequential(*copy.deepcopy(list(vgg_decoder_relu5_1.children())))
        else:
            raise ValueError('level should be between 1~5')

        if pretrained_path is not None:
            self.net.load_state_dict(torch.load(pretrained_path, map_location=lambda storage, loc: storage))

    def forward(self, x):
        return self.net(x)
