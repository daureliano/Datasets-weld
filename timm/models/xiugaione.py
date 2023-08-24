import torch
import torch.nn as nn
import torch.nn.init as init
from torch.hub import load_state_dict_from_url
from typing import Any

# 20230313尝试自己仿照修改的程序进行仿写，添加深度可分离模块depthwiseconv_mix，添加残差模块Resblock_body。
__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']  # 对参数空间进行了探索，主要研究了压缩比和3∗3卷积比例的影响

model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth',
}


# 深度可分离卷积    深度可分离卷积应用于下面的Fire模块
class depthwiseconv_mix(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding):
        super(depthwiseconv_mix, self).__init__()
        self.depth_conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding, groups=in_ch)  # 逐通道卷积
        self.point_conv = nn.Conv2d(in_channels=out_ch,  # 逐点卷积
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=1,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


# 残差块  第一个残差模块
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblock_body, self).__init__()
        self.blocks_conv = nn.Sequential(
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),  # 主程序squeezenet中位于前两个池化层中间的三个Fire模块添加到此处
            Fire(128, 32, 128, 128),
        )

    def forward(self, x):
        x = self.blocks_conv(x)
        return x


# 残差模块 第二个残差模块
class Resblock_body1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblock_body1, self).__init__()
        self.blocks_conv = nn.Sequential(
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),  # 主程序squeezenet中位于后两个池化层中间的四个Fire模块添加到此处
            Fire(384, 64, 256, 256), )

    def forward(self, x):
        x = self.blocks_conv(x)
        return x


#  新的网络架构Fire Module，深度混合可分离卷积，添加了residual
class Fire(nn.Module):  # 新的网络架构Fire Module，深度混合可分离卷积

    def __init__(
            self,
            inplanes: int,  # 输入向量   96
            squeeze_planes: int,  # 输出通道  16
            expand1x1_planes: int,  # 64
            expand3x3_planes: int,  # 64
    ) -> None:
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)  # 56*56*16
        self.squeeze_bn = nn.BatchNorm2d(squeeze_planes)  # 加入bn层
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)  # 56*56*64
        self.expand1x1_bn = nn.BatchNorm2d(expand1x1_planes)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = depthwiseconv_mix(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)  # 3*3深度可分离卷积
        self.expand3x3_bn = nn.BatchNorm2d(expand3x3_planes)  # 加入bn层
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_bn(self.expand1x1_activation(self.expand1x1(x))),
            self.expand3x3_bn(self.expand3x3_activation(self.expand3x3(x)))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(
            self,
            version: str = '1_0',
            num_classes: int = 6  # 分类的类别个数1000
    ) -> None:
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes

        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),  # 111*111*96  conv1
                nn.ReLU(inplace=True),  # 111*111*96
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 56*56*96  maxpool1
                # Fire(96, 16, 64, 64),  # Fire2
                # Fire(128, 16, 64, 64),  # Fire3
                # Fire(128, 32, 128, 128),  # Fire4
                Resblock_body(96, 256),  # 上一个池化层输出的是96，下一个池化层输出的是256
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # maxpool4 27*27*256
                # Fire(256, 32, 128, 128),  # Fire5
                # Fire(256, 48, 192, 192),  # Fire6
                # Fire(384, 48, 192, 192),  # Fire7
                # Fire(384, 64, 256, 256),  # Fire8
                Resblock_body1(256, 512),  # 上一个池化层输出的是256，下一个池化层输出的是512
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # ceil_mode=True会对池化结果进行向上取整而不是向下取整 maxpool8
                Fire(512, 64, 256, 256),  # Fire9
            )
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # isinstance() 函数来判断一个对象是否是一个已知的类型
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)  # 用正态分布N(mean, std2)的值填充输入张量。
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


def _squeezenet(version: str, pretrained: bool, progress: bool, **kwargs: Any) -> SqueezeNet:
    model = SqueezeNet(version, **kwargs)
    if pretrained:
        arch = 'squeezenet' + version
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def squeezenet1_0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SqueezeNet:
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_0', pretrained, progress, **kwargs)


def squeezenet1_1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SqueezeNet:
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_1', pretrained, progress, **kwargs)
