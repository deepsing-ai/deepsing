import torch
import torch.nn as nn

# code from https://github.com/irasin/Pytorch_WCT


normalised_vgg_relu5_1 = nn.Sequential(
	nn.Conv2d(3, 3, 1),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(3, 64, 3),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64, 64, 3),
	nn.ReLU(),
	nn.MaxPool2d(2, ceil_mode=True),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64, 128, 3),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128, 128, 3),
	nn.ReLU(),
	nn.MaxPool2d(2, ceil_mode=True),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128, 256, 3),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256, 256, 3),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256, 256, 3),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256, 256, 3),
	nn.ReLU(),
	nn.MaxPool2d(2, ceil_mode=True),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256, 512, 3),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512, 512, 3),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512, 512, 3),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512, 512, 3),
	nn.ReLU(),
	nn.MaxPool2d(2, ceil_mode=True),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512, 512, 3),
	nn.ReLU()
	)


class NormalisedVGG(nn.Module):
	"""
	VGG reluX_1(X = 1, 2, 3, 4, 5) can be obtained by slicing the follow vgg5_1 model.

	Sequential(
	(0): Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1))
	(1): ReflectionPad2d((1, 1, 1, 1))
	(2): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
	(3): ReLU() # relu1_1
	(4): ReflectionPad2d((1, 1, 1, 1))
	(5): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
	(6): ReLU()
	(7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
	(8): ReflectionPad2d((1, 1, 1, 1))
	(9): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
	(10): ReLU() # relu2_1
	(11): ReflectionPad2d((1, 1, 1, 1))
	(12): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
	(13): ReLU()
	(14): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
	(15): ReflectionPad2d((1, 1, 1, 1))
	(16): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
	(17): ReLU() # relu3_1
	(18): ReflectionPad2d((1, 1, 1, 1))
	(19): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
	(20): ReLU()
	(21): ReflectionPad2d((1, 1, 1, 1))
	(22): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
	(23): ReLU()
	(24): ReflectionPad2d((1, 1, 1, 1))
	(25): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
	(26): ReLU()
	(27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
	(28): ReflectionPad2d((1, 1, 1, 1))
	(29): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1))
	(30): ReLU()# relu4_1
	(31): ReflectionPad2d((1, 1, 1, 1))
	(32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1))
	(33): ReLU()
	(34): ReflectionPad2d((1, 1, 1, 1))
	(35): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1))
	(36): ReLU()
	(37): ReflectionPad2d((1, 1, 1, 1))
	(38): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1))
	(39): ReLU()
	(40): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
	(41): ReflectionPad2d((1, 1, 1, 1))
	(42): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1))
	(43): ReLU() # relu5_1
	)
	"""
	def __init__(self, pretrained_path='vgg_normalised_conv5_1.pth'):
		super().__init__()
		self.net = normalised_vgg_relu5_1
		if pretrained_path is not None:
			self.net.load_state_dict(torch.load(pretrained_path, map_location=lambda storage, loc: storage))

	def forward(self, x, target):
		if target == 'relu1_1':
			return self.net[:4](x)
		elif target == 'relu2_1':
			return self.net[:11](x)
		elif target == 'relu3_1':
			return self.net[:18](x)
		elif target == 'relu4_1':
			return self.net[:31](x)
		elif target == 'relu5_1':
			return self.net(x)
		else:
			raise ValueError(f'target should be in ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"] but not {target}')
