import torch
import torch.nn as nn
import torch.nn.init as init
from torch.hub import load_state_dict_from_url
from typing import Any

__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']   #对参数空间进行了探索，主要研究了压缩比和3∗3卷积比例的影响

model_urls = { #squeezenet1_0 使用其他方法对提出的SqeezeNet模型进行进一步压缩
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth',
}

#残差块
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblock_body, self).__init__()

        self.blocks_conv = nn.Sequential(
            Fire(96, 16, 64, 64, 0),
            Fire(128, 16, 64, 64, 1),
            Fire(128, 32, 128, 128, 0),)

    def forward(self, x):
        x = self.blocks_conv(x)
        #x = torch.cat([x1, x], dim=1)
        return x

class Resblock_body1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblock_body1, self).__init__()
        self.blocks_conv = nn.Sequential(
            Fire(256, 32, 128, 128, 1),
            Fire(256, 48, 192, 192, 0),
            Fire(384, 48, 192, 192, 1),
            Fire(384, 64, 256, 256, 0),)

    def forward(self, x):
        x = self.blocks_conv(x)
        #x = torch.cat([x1, x], dim=1)
        return x

#深度可分离卷积
class depthwiseconv_mix(nn.Module):
    def __init__(self, in_ch, out_ch,kernel_size,padding):
        super(depthwiseconv_mix, self).__init__()
        self.depth_conv = nn.Conv2d(in_ch, in_ch, kernel_size, padding=1,groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class Fire(nn.Module):   #新的网络架构Fire Module，通过减少参数来进行模型压缩

    def __init__(
        self,
        inplanes: int,        #输入向量96
        squeeze_planes: int,      #输出通道16
        expand1x1_planes: int,      #64
        expand3x3_planes: int,     #64
        residual
    ) -> None:
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.residual = residual
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm2d(squeeze_planes)
        self.squeeze_activation = nn.ReLU(inplace=True)   #inplace=True参数可以在原始的输入上直接进行操作，不会再
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,  #为输出分配额外的内存，可以节省一部分内存，但同时也会破坏原始的输入
                                   kernel_size=1)
        self.expand1x1_bn = nn.BatchNorm2d(expand1x1_planes)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        #self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,kernel_size=3, padding=1)
        self.expand3x3 = depthwiseconv_mix(squeeze_planes, expand3x3_planes,kernel_size=3, padding=1)  # 3*3深度可分离卷积
        self.expand3x3_bn = nn.BatchNorm2d(expand3x3_planes)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            identity = x
        x =self.squeeze_bn(self.squeeze_activation(self.squeeze(x)))
        x=torch.cat([    #torch.cat()将expand1x1_activation和 expand3x3_activation这两个维度相同的输出张量连接在一起
            self.expand1x1_bn(self.expand1x1_activation(self.expand1x1(x))),
            self.expand3x3_bn(self.expand3x3_activation(self.expand3x3(x)))
        ], 1)   #dim=1，即按照列连接，最终得到若干行
        if self.residual:
            x = x + identity
        return x

"""（1）使用1∗11∗1卷积代替3∗33∗3 卷积：参数减少为原来的1/9
（2）减少输入通道数量：这一部分使用squeeze layers来实现
（3）将欠采样操作延后，可以给卷积层提供更大的激活图：更大的激活图保留了更多的信息，可以提供更高的分类准确率
其中，（1）和（2）可以显著减少参数数量，（3）可以在参数数量受限的情况下提高准确率
"""
class SqueezeNet(nn.Module):

    def __init__(
        self,
        version: str = '1_0',
        num_classes: int = 6  #分类的类别个数1000
    ) -> None:
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes

        if version == '1_0':
            self.features = nn.Sequential(      #self.features定义了主要的网络层
                nn.Conv2d(3, 96, kernel_size=7, stride=2),   #111*111*96
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),    #55*55*96
                # Fire(96, 16, 64, 64,0),
                # Fire(128, 16, 64, 64,1),
                # Fire(128, 32, 128, 128,0),
                Resblock_body(96, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), #27*27*256
                # Fire(256, 32, 128, 128,1),
                # Fire(256, 48, 192, 192,0),
                # Fire(384, 48, 192, 192,1),
                # Fire(384, 64, 256, 256,0),
                Resblock_body1(256, 512),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),   #ceil_mode=True会对池化结果进行向上取整而不是向下取整
                Fire(512, 64, 256, 256, 1),
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
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
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
            if isinstance(m, nn.Conv2d):  #isinstance() 函数来判断一个对象是否是一个已知的类型
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)  #用正态分布N(mean, std2)的值填充输入张量。
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
