import torch
import torch.nn as nn
import math


class Fire(nn.Module):  # 新的网络架构Fire Module，通过减少参数来进行模型压缩

    def __init__(
            self,
            inplanes: int,  # 输入向量
            squeeze_planes: int,  # 后面会直接规定挤压层的通道数为16
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


if __name__ == '__main__':
    input = torch.rand(1, 3, 224, 224)  # 构造输入层  batch, channel , height , width
    print('原始输入层大小:', input.shape)
    m = nn.Conv2d(3, 96, kernel_size=7, stride=2)  # in_channel， out_channel, kernel_size, stride
    print(m)
    y = m(input)
    print('经过卷积层后的输出大小:', y.shape)  # 检查经过卷积层后的输出大小
    # torch.Size([1, 96, 109, 109])
    n = nn.ReLU(inplace=True)
    n_out = n(y)
    print('经过ReLU后的输出大小:', n_out.shape)  # 检查经过ReLU后的输出大小
    # torch.Size([1, 96, 109, 109])
    max1_c = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
    max1_out = max1_c(n_out)
    print('经过第一个MaxPool后的输出大小:', max1_out.shape)  # 检查经过MaxPool后的输出大小
    # torch.Size([1, 96, 54, 54])   batch, channel , height , width
    fire1_n = Fire(96, 16, 64, 64)  # 这个96来自上一层输出的channel
    fire1_o = fire1_n(max1_out)
    print('经过第1个fire后的输出大小:', fire1_o.shape)  # 检查经过第1个fire后的输出大小
    # torch.Size([1, 128, 54, 54])  batch, channel , height , width
    fire2_n = Fire(128, 16, 64, 64)  # Fire的128来自上一层输出的channel
    fire2_o = fire2_n(fire1_o)
    print('经过第2个fire后的输出大小:', fire2_o.shape)  # 检查经过第2个fire后的输出大小
    # torch.Size([1, 128, 54, 54])  batch, channel , height , width
    fire3_n = Fire(128, 32, 128, 128)  # Fire的第一个128来自上一层输出的channel
    fire3_o = fire3_n(fire2_o)
    print('经过第3个fire后的输出大小:', fire3_o.shape)  # 检查经过第3个fire后的输出大小
    # torch.Size([1, 256, 54, 54])  batch, channel , height , width
    max2_c = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
    max2_o = max2_c(fire3_o)
    print('经过第2个MaxPool后的输出大小:', max2_o.shape)  # 检查经过MaxPool后的输出大小
    # torch.Size([1, 256, 27, 27])  batch, channel , height , width
    fire4_n = Fire(256, 32, 128, 128)  # Fire的第一个256来自上一层输出的channel
    fire4_o = fire4_n(max2_o)
    print('经过第4个fire后的输出大小:', fire4_o.shape)  # 检查经过第4个fire后的输出大小
    # torch.Size([1, 256, 27, 27]) batch, channel , height , width
    fire5_n = Fire(256, 48, 192, 192)  # Fire的第一个256来自上一层输出的channel
    fire5_o = fire5_n(fire4_o)
    print('经过第5个fire后的输出大小:', fire5_o.shape)  # 检查经过第5个fire后的输出大小
    # torch.Size([1, 384, 27, 27])  batch, channel , height , width
    fire6_n = Fire(384, 48, 192, 192)
    fire6_o = fire6_n(fire5_o)
    print('经过第6个fire后的输出大小:', fire6_o.shape)  # 检查经过第6个fire后的输出大小
    # torch.Size([1, 384, 27, 27]) batch, channel , height , width
    eca_n = eca_block(384, b=1, gama=2)
    eca_o = eca_n(fire6_o)
    print('经过eca模块后的输出大小:', eca_o.shape)   # 检查经过经过eca模块后的输出大小







