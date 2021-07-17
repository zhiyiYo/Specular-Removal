# coding:utf-8
import torch
import numpy as np
from torch import nn
from PIL import Image
from torchvision import transforms as T


class EncoderBlock(nn.Module):
    """ 编码器卷积块 """

    def __init__(self, in_channels: int, out_channel: int, kernel_size=3, padding=0):
        """
        Parameters
        ----------
        in_channels: int
            输入通道数

        out_channels: int
            输出通道数

        kernel_size: int
            卷积核大小

        padding: int
            卷积的 padding 大小
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self.features(x)


class DecoderBlock(nn.Module):
    """ 解码器卷积块 """

    def __init__(self, in_channels: int, out_channel: int, kernel_size, padding=0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channel, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.features(x)


class SDNet(nn.Module):
    """ 高光检测网络 """

    def __init__(self, in_channels=3):
        super().__init__()
        self.encoder1 = EncoderBlock(in_channels, 8, 3, padding=1)
        self.encoder2 = EncoderBlock(8, 16, 3, padding=1)
        self.encoder3 = EncoderBlock(16, 32, 3, padding=1)
        self.decoder3 = DecoderBlock(32, 16, 3, padding=1)
        self.decoder2 = DecoderBlock(16, 8, 3, padding=1)
        self.decoder1 = DecoderBlock(8, 4, 3, padding=1)
        self.decoder0 = nn.Sequential(
            nn.Conv2d(4, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x: Tensor of shape `(N, C, H, W)`
        """
        encoder_out1 = self.encoder1(x)
        encoder_out2 = self.encoder2(encoder_out1)
        encoder_out3 = self.encoder3(encoder_out2)
        decoder_out3 = self.decoder3(encoder_out3)
        print(encoder_out2.shape, decoder_out3.shape)
        decoder_out2 = self.decoder2(decoder_out3 + encoder_out2)
        decoder_out1 = self.decoder1(decoder_out2 + encoder_out1)
        decoder_out0 = self.decoder0(decoder_out1)
        return decoder_out0

    def predict(self, image: Image.Image, use_gpu=True):
        """ 预测高光区域

        Parameters
        ----------
        image: ~PIL.Image.Image
            PIL 图像

        use_gpu: bool
            是否使用 GPU

        Return
        ------
        mask: ~PIL.Image.Image
            蒙版灰度图像
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = T.ToTensor()(image).unsqueeze(0)  # type:torch.Tensor
        mask = self(image.to('cuda:0' if use_gpu else 'cpu'))
        mask = T.ToPILImage()(mask.squeeze().to('cpu'))
        return mask
