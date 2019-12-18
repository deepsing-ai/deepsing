import torch.nn as nn
from stylization_utils.normalisedVGG import NormalisedVGG
from stylization_utils.VGGdecoder import Decoder
from stylization_utils.feature_transformer import whiten_and_color

# code from https://github.com/irasin/Pytorch_WCT


class SingleLevelAE(nn.Module):
    def __init__(self, level, pretrained_path_dir='stylization_model'):
        super().__init__()
        self.level = level
        self.encoder = NormalisedVGG(f'{pretrained_path_dir}/vgg_normalised_conv5_1.pth')
        self.decoder = Decoder(level, f'{pretrained_path_dir}/decoder_relu{level}_1.pth')

    def forward(self, content_image, style_image, alpha):
        content_feature = self.encoder(content_image, f'relu{self.level}_1')
        style_feature = self.encoder(style_image, f'relu{self.level}_1')
        res = whiten_and_color(content_feature, style_feature, alpha)
        res = self.decoder(res)
        return res


class MultiLevelAE(nn.Module):
    def __init__(self, pretrained_path_dir='stylization_model'):
        super().__init__()
        self.encoder = NormalisedVGG(f'{pretrained_path_dir}/vgg_normalised_conv5_1.pth')
        self.decoder1 = Decoder(1, f'{pretrained_path_dir}/decoder_relu1_1.pth')
        self.decoder2 = Decoder(2, f'{pretrained_path_dir}/decoder_relu2_1.pth')
        self.decoder3 = Decoder(3, f'{pretrained_path_dir}/decoder_relu3_1.pth')
        self.decoder4 = Decoder(4, f'{pretrained_path_dir}/decoder_relu4_1.pth')
        # self.decoder5 = Decoder(5, f'{pretrained_path_dir}/decoder_relu5_1.pth')

    def transform_level(self, content_image, style_image, alpha, level):
        content_feature = self.encoder(content_image, f'relu{level}_1')
        # print(content_feature.shape)
        style_feature = self.encoder(style_image, f'relu{level}_1')
        res = whiten_and_color(content_feature, style_feature, alpha)
        return getattr(self, f'decoder{level}')(res)

    def forward(self, content_image, style_image, alpha=1):
        # r5 = self.transform_level(content_image, style_image, alpha, 5)
        r4 = self.transform_level(content_image, style_image, alpha, 4)
        r3 = self.transform_level(r4, style_image, alpha, 3)
        r2 = self.transform_level(r3, style_image, alpha, 2)
        r1 = self.transform_level(r2, style_image, alpha, 1)

        return r1
