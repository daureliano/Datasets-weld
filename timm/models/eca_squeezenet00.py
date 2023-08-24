import torch
import torch.nn as nn
import torch.nn.init as init
import math
from torch.hub import load_state_dict_from_url
from typing import Any

__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']  # 对参数空间进行了探索，主要研究了压缩比和3∗3卷积比例的影响

model_urls = {  # squeezenet1_0 使用其他方法对提出的SqeezeNet模型进行进一步压缩
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth',
}


# 定义ECANet的类
class eca_block(nn.Module):
    # 初始化, in_channel代表特征图的输入通道数, b和gama代表公式中的两个系数
    def __init__(self, in_channel, b=1, gama=2):
        # 继承父类初始化
        super(eca_block, self).__init__()

        # 根据输入通道数自适应调整卷积核大小
        kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
        # 如果卷积核大小是奇数，就使用它
        if kernel_size % 2:
            kernel_size = kernel_size
        # 如果卷积核大小是偶数，就把它变成奇数
        else:
            kernel_size = kernel_size + 1

        # 卷积时，为例保证卷积前后的size不变，需要0填充的数量
        padding = kernel_size // 2

        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # 1D卷积，输入和输出通道数都=1，卷积核大小是自适应的
        # 这个1维卷积需要好好了解一下机制，这是改进SENet的重要不同点
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                              bias=False, padding=padding)
        # sigmoid激活函数，权值归一化
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 获得输入图像的shape
        b, c, h, w = inputs.shape

        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        x = self.avg_pool(inputs)
        # 维度调整，变成序列形式 [b,c,1,1]==>[b,1,c]
        x = x.view([b, 1, c])  # 这是为了给一维卷积
        # 1D卷积 [b,1,c]==>[b,1,c]
        x = self.conv(x)
        # 权值归一化
        x = self.sigmoid(x)
        # 维度调整 [b,1,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])

        # 将输入特征图和通道权重相乘[b,c,h,w]*[b,c,1,1]==>[b,c,h,w]
        outputs = x * inputs
        return outputs


class Fire(nn.Module):  # 新的网络架构Fire Module，通过减少参数来进行模型压缩

    def __init__(
            self,
            inplanes: int,  # 输入向量96
            squeeze_planes: int,  # 输出通道16
            expand1x1_planes: int,  # 64
            expand3x3_planes: int,  # 64
    ) -> None:
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)  # inplace=True参数可以在原始的输入上直接进行操作，不会再
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,  # 为输出分配额外的内存，可以节省一部分内存，但同时也会破坏原始的输入
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([  # torch.cat()将expand1x1_activation和 expand3x3_activation这两个维度相同的输出张量连接在一起
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)  # dim=1，即按照列连接，最终得到若干行


"""（1）使用1∗11∗1卷积代替3∗33∗3 卷积：参数减少为原来的1/9
（2）减少输入通道数量：这一部分使用squeeze layers来实现
（3）将欠采样操作延后，可以给卷积层提供更大的激活图：更大的激活图保留了更多的信息，可以提供更高的分类准确率
其中，（1）和（2）可以显著减少参数数量，（3）可以在参数数量受限的情况下提高准确率
"""


class SqueezeNet(nn.Module):

    def __init__(
            self,
            version: str = '1_0',
            num_classes: int = 6  # 分类的类别个数1000  class修改为6 放入我的图片仍然能够运行
    ) -> None:
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes

        if version == '1_0':
            self.features = nn.Sequential(  # self.features定义了主要的网络层
                nn.Conv2d(3, 96, kernel_size=7, stride=2),  # 111*111*96输出torch.Size([1, 96, 109, 109])
                nn.ReLU(inplace=True),  # torch.Size([1, 96, 109, 109])
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 55*55*96输出torch.Size([1, 96, 54, 54])
                Fire(96, 16, 64, 64),  # 输出torch.Size([1, 128, 54, 54])  batch, channel , height , width
                Fire(128, 16, 64, 64),  # 输出torch.Size([1, 128, 54, 54])  batch, channel , height , width
                Fire(128, 32, 128, 128),    # 输出torch.Size([1, 256, 27, 27])  batch, channel , height , width
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),   # 输出torch.Size([1, 256, 27, 27])
                Fire(256, 32, 128, 128),  # 输出torch.Size([1, 256, 27, 27]) batch, channel , height , width
                Fire(256, 48, 192, 192),  # 输出torch.Size([1, 384, 27, 27])  batch, channel , height , width
                Fire(384, 48, 192, 192),  # 输出torch.Size([1, 384, 27, 27]) batch, channel , height , width
                eca_block(384, b=1, gama=2),  # 经过eca模块后的输出大小: torch.Size([1, 384, 27, 27])
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # ceil_mode=True会对池化结果进行向上取整而不是向下取整
                Fire(512, 64, 256, 256),
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
