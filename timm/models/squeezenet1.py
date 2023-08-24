import torch
import torch.nn as nn
import torch.nn.init as init

# from .common import conv1x1, depthwise_conv3x3, conv1x1_block, conv3x3_block, ChannelShuffle, SEBlock, dwconv3x3_block
try:
    from torch.hub import load_state_dict_from_url  # 加载模型
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']

model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


def depthwise_conv3x3(channels, stride):
    return nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=stride, padding=1,
                     groups=channels, bias=False)


class depthwiseconv(nn.Module):  # mobilenet中，深度可分离卷积
    def __init__(self, in_ch, out_ch):
        super(depthwiseconv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    # Depthwise Convolution完成后的Feature map数量与输入层的通道数相同，无法扩展Feature map
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,  # Pointwise Convolution有几个卷积核就有几个输出Feature map
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        # print(out.shape)
        out = channel_shuffle(out, out.shape[1])  # 达到不同组特征通信的目的
        out = self.point_conv(out)
        return out


class depthwiseconv1(nn.Module):  # stride不同，为2
    def __init__(self, in_ch, out_ch):
        super(depthwiseconv1, self).__init__()
        self.depth_conv1 = nn.Conv2d(in_channels=in_ch,
                                     out_channels=in_ch,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     groups=in_ch)
        self.point_conv1 = nn.Conv2d(in_channels=in_ch,
                                     out_channels=out_ch,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     groups=1)

    def forward(self, input):
        out = self.depth_conv1(input)
        out = self.point_conv1(out)
        return out


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()  # x.data.size() 得到X的四个维度
    channels_per_group = num_channels // groups  # 每组几个通道
    # 1将Feature Map展开成g*n*w*h的思维矩阵 2沿着尺寸为g*n*s的矩阵的g轴和n轴进行转置 3将g轴和n轴进行平铺后得到洗牌之后的Feature Map 4进行组内1*1卷积
    # reshape   b,c,h,w=====>b,g,c_per,h,w
    x = x.view(batchsize, groups,
               channels_per_group, height, width)  # view第0，3，4都是x.data.size() 得到的，没有发生变化      将channels进行reshape

    x = torch.transpose(x, 1, 2).contiguous()  # contiguous()回到与维度对应的存储状态

    # flatten
    x = x.view(batchsize, -1, height, width)  # 相当于逆操作，其他三维都不变化，因此把中间的第1，2维reshape为一个向量

    return x


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6(inplace)


def _SplitChannels(channels, num_groups):  # 通道分组   mixnet 根据组数对输入通道分组   num_groups=3
    split_channels = [channels // num_groups for _ in range(num_groups)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels


class MDConv(nn.Module):
    def __init__(self, channels, kernel_size, stride):
        super(MDConv, self).__init__()

        self.num_groups = len(kernel_size)  # 最终kernel_size加载出来的个数    3，5，7=3
        self.split_channels = _SplitChannels(channels, self.num_groups)  # 每组通道数

        self.mixed_depthwise_conv = nn.ModuleList()
        for i in range(self.num_groups):  # 0，1，2
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
        # print(out.shape)
        # out = channel_shuffle(out, out.shape[1])
        out = self.point_conv(out)
        return out


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, kernel_list, expand3x3_planes, residual):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.residual = residual  # 残差
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm2d(squeeze_planes)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_bn = nn.BatchNorm2d(expand1x1_planes)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        # self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
        #                           kernel_size=3, padding=1)
        self.expand3x3 = depthwiseconv_mix(squeeze_planes, kernel_list, expand3x3_planes)
        self.expand3x3_bn = nn.BatchNorm2d(expand3x3_planes)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.residual:
            identity = x  # shortcut连接
        x = self.squeeze_bn(self.squeeze_activation(self.squeeze(x)))
        x = channel_shuffle(torch.cat([
            self.expand1x1_bn(self.expand1x1_activation(self.expand1x1(x))),
            self.expand3x3_bn(self.expand3x3_activation(self.expand3x3(x)))
        ], 1), 2)
        if self.residual:
            x = x + identity
        return x


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
            # self.split_conv0 = BasicConv(in_channels, out_channels//2, 1)
            self.split_conv1 = BasicConv(in_channels, in_channels // 2, 1)  # 得到x

            self.blocks_conv = nn.Sequential(
                Fire(48, 8, 32, [3, 5, 7], 32, 0),
                Fire(64, 8, 32, [3, 5, 7], 32, 1),
                Fire(64, 16, 80, [3, 5, 7], 80, 0),
                # BasicConv(out_channels//2, out_channels//2, 1)
            )
            # self.concat_conv = BasicConv(out_channels, out_channels, 1)

    def forward(self, x):
        if self.first:
            x = self.features(x)
            return x
        # x0 = self.split_conv0(x)
        else:
            x1 = self.split_conv1(x)
            x1 = self.blocks_conv(x1)

            x = torch.cat([x1, x], dim=1)
            # x = self.concat_conv(x)

            return x


class Resblock_body1(nn.Module):
    def __init__(self, in_channels, out_channels, first):
        super(Resblock_body1, self).__init__()

        if first:
            Fire(256, 32, 128, [3, 5, 7], 128, 1),
            Fire(256, 48, 192, [3, 5, 7], 192, 0),
            Fire(384, 48, 192, [3, 5, 7], 192, 1),
            Fire(384, 64, 256, [3, 5, 7], 256, 0),
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
            # self.concat_conv = BasicConv(out_channels, out_channels, 1)

    def forward(self, x):

        # x0 = self.split_conv0(x)

        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)

        x = torch.cat([x1, x], dim=1)
        # x = self.concat_conv(x)

        return x


class SqueezeNet(nn.Module):

    def __init__(self, version='1_0', num_classes=6):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),  # 111*111*96
                # nn.Conv2d(3, 16, kernel_size=7, stride=2),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(16, 96, kernel_size=3, stride=1, padding=1, groups=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 等于True，计算输出信号大小的时候，会使用向上取整代替默认的向下取整的操作
                # 56*56*96
                # Fire(96, 16, 64, [3, 5, 7], 64, 0),
                # Fire(128, 16, 64, [3, 5, 7], 64, 1),
                # Fire(128, 32, 128, [3, 5, 7], 128, 0),
                Resblock_body(96, 256, first=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                # Fire(256, 32, 128, [3, 5, 7], 128, 1),
                # Fire(256, 48, 192, [3, 5, 7], 192, 0),
                # Fire(384, 48, 192, [3, 5, 7], 192, 1),
                # Fire(384, 64, 256, [3, 5, 7], 256, 0),
                Resblock_body1(256, 512, first=False),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, [3, 5, 7], 256, 1),
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
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # 随机失活
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


def _squeezenet(version, pretrained, progress, **kwargs):
    model = SqueezeNet(version, **kwargs)
    if pretrained:
        arch = 'squeezenet' + version
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def squeezenet1_0(pretrained=False, progress=True, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_0', pretrained, progress, **kwargs)


def squeezenet1_1(pretrained=False, progress=True, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_1', pretrained, progress, **kwargs)



