import torch
import torch.nn as nn
import torch.nn.init as init
from torch.hub import load_state_dict_from_url
from typing import Any

__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']  # 对参数空间进行了探索，主要研究了压缩比和3∗3卷积比例的影响

model_urls = {  # squeezenet1_0 使用其他方法对提出的SqeezeNet模型进行进一步压缩
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth',
}


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


# 残差块
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels, first):
        super(Resblock_body, self).__init__()
        self.first = first
        if first:
            self.features = nn.Sequential(
                Fire(96, 16, 64, [3, 5, 7], 64, 0),
                Fire(128, 16, 64, [3, 5, 7], 64, 1),
                Fire(128, 32, 128, [3, 5, 7], 128, 0), )
        else:
            self.split_conv1 = BasicConv(in_channels, in_channels // 2, 1)
            self.blocks_conv = nn.Sequential(
                Fire(48, 8, 32, [3, 5, 7], 32, 0),
                Fire(64, 8, 32, [3, 5, 7], 32, 1),
                Fire(64, 16, 80, [3, 5, 7], 80, 0), )

    def forward(self, x):
        if self.first:
            x = self.features(x)
            return x
        else:
            x1 = self.split_conv1(x)
            x1 = self.blocks_conv(x1)
            x = torch.cat([x1, x], dim=1)
            # x = self.blocks_conv(x)
            # x = torch.cat([x1, x], dim=1)
            return x


class Resblock_body1(nn.Module):
    def __init__(self, in_channels, out_channels, first):
        super(Resblock_body1, self).__init__()
        self.first = first
        if first:
            self.features = nn.Sequential(
                Fire(256, 32, 128, [3, 5, 7], 128, 1),
                Fire(256, 48, 192, [3, 5, 7], 192, 0),
                Fire(384, 48, 192, [3, 5, 7], 192, 1),
                Fire(384, 64, 256, [3, 5, 7], 256, 0), )
        else:
            # self.split_conv0 = BasicConv(in_channels, out_channels//2, 1)
            self.split_conv1 = BasicConv(in_channels, in_channels // 2, 1)
            self.blocks_conv = nn.Sequential(
                Fire(128, 16, 64, [3, 5, 7], 64, 1),
                Fire(128, 24, 96, [3, 5, 7], 96, 0),
                Fire(192, 24, 96, [3, 5, 7], 96, 1),
                Fire(192, 24, 128, [3, 5, 7], 128, 0),
                # BasicConv(out_channels//2, out_channels//2, 1)
            )

    def forward(self, x):
        # x = self.blocks_conv(x)
        # #x = torch.cat([x1, x], dim=1)
        # return x
        if self.first:
            x = self.features(x)
            return x
        else:
            x1 = self.split_conv1(x)
            x1 = self.blocks_conv(x1)
            x = torch.cat([x1, x], dim=1)
            # x = self.blocks_conv(x)
            # x = torch.cat([x1, x], dim=1)
            return x


# 深度可分离卷积
class depthwiseconv_mix(nn.Module):
    def __init__(self, in_ch, kernel_list, out_ch):
        super(depthwiseconv_mix, self).__init__()
        self.depth_conv = MDConv(in_ch, kernel_list, 1)
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


# 3*3depthwise替换为mixconv
def _SplitChannels(channels, num_groups):  # 通道分组   mixnet 根据组数对输入通道分组   num_groups=3
    split_channels = [channels // num_groups for _ in range(num_groups)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels


class MDConv(nn.Module):
    def __init__(self, channels, kernel_size, stride):
        super(MDConv, self).__init__()

        self.num_groups = len(kernel_size)  # 最终kernel_size加载出来的个数    3，5，7=3
        self.split_channels = _SplitChannels(channels, self.num_groups)

        self.mixed_depthwise_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.mixed_depthwise_conv.append(nn.Conv2d(  # append在列表中追加元素
                self.split_channels[i],
                self.split_channels[i],
                kernel_size[i],
                stride=stride,
                padding=kernel_size[i] // 2,
                groups=self.split_channels[i],
                bias=False
            ))

    def forward(self, x):
        if self.num_groups == 1:  # 当组数为一时就是普通的深度可分离卷积
            return self.mixed_depthwise_conv[0](x)

        x_split = torch.split(x, self.split_channels, dim=1)  # torch.split将x分成快结构   self.split_channels需要切分的大小，dim切分维度
        x = [conv(t) for conv, t in zip(self.mixed_depthwise_conv, x_split)]
        for i in range(len(x)):
            x[i] = channel_shuffle(x[i], self.split_channels[i])
            # print(i, self.split_channels[i])
        x = torch.cat(x, dim=1)
        return x


# 通道混洗
def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class Fire(nn.Module):  # 新的网络架构Fire Module，通过减少参数来进行模型压缩

    def __init__(
            self,
            inplanes: int,  # 输入向量96
            squeeze_planes: int,  # 输出通道16
            expand1x1_planes: int,  # 64
            kernel_list,
            expand3x3_planes: int,  # 64
            residual
    ) -> None:
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.residual = residual
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm2d(squeeze_planes)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_bn = nn.BatchNorm2d(expand1x1_planes)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        # self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,kernel_size=3, padding=1)
        self.expand3x3 = depthwiseconv_mix(squeeze_planes, kernel_list, expand3x3_planes)  # 3*3深度可分离卷积
        self.expand3x3_bn = nn.BatchNorm2d(expand3x3_planes)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            identity = x
        x = self.squeeze_bn(self.squeeze_activation(self.squeeze(x)))
        x = channel_shuffle(torch.cat([
            self.expand1x1_bn(self.expand1x1_activation(self.expand1x1(x))),
            self.expand3x3_bn(self.expand3x3_activation(self.expand3x3(x)))
        ], 1), 2)  # dim=1，即按照列连接，最终得到若干行
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
            num_classes: int = 6  # 分类的类别个数1000  共有六个类别
    ) -> None:
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        if version == '1_0':
            self.features = nn.Sequential(  # self.features定义了主要的网络层   输入尺寸：224*224*3
                nn.Conv2d(3, 96, kernel_size=7, stride=2),  # 111*111*96 卷积层 96是通道数，由原先的三通道变成现在的96
                nn.ReLU(inplace=False),   # 激活函数 取值为False时不改变原输入
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 55*55*96  最大池化层
                # Fire(96, 16, 64, 64,0),
                # Fire(128, 16, 64, 64,1),
                # Fire(128, 32, 128, 128,0),
                Resblock_body(96, 256, first=False),  # False 第一个残差块
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 27*27*256
                # Fire(256, 32, 128, 128,1),
                # Fire(256, 48, 192, 192,0),
                # Fire(384, 48, 192, 192,1),
                # Fire(384, 64, 256, 256,0),
                Resblock_body1(256, 512, first=False),  # True 第二个残差块
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # ceil_mode=True会对池化结果进行向上取整而不是向下取整
                Fire(512, 64, 256, [3, 5, 7], 256, 1),  # 20230222将其注释 错误如常
            )
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, [3, 5, 7], 64, 0),
                Fire(128, 16, 64, [3, 5, 7], 64, 1),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, [3, 5, 7], 128, 0),
                Fire(256, 32, 128, [3, 5, 7], 128, 1),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, [3, 5, 7], 192, 0),
                Fire(384, 48, 192, [3, 5, 7], 192, 1),
                Fire(384, 64, 256, [3, 5, 7], 256, 0),
                Fire(512, 64, 256, [3, 5, 7], 256, 1),
            )
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)  # 将在下面的分类器中使用final_conv
        self.classifier = nn.Sequential(
            # nn.Linear(512,25088),  # 20230222添加 ，测试未果。错误如常
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
        # print(x.shape)
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
