# -- coding: utf-8 --
# 输出模型，检查逻辑，
import torch
import torchvision
# from timm.models.squeezenet2 import squeezenet1_0  # 加载文件中的模型
from timm.models.eca_squeezenet00 import squeezenet1_0  # 加载文件中的模型

model = squeezenet1_0()  # 模型实例化
input = torch.rand(1, 3, 224, 224)  # 构造输入层
output = model(input)  # 前向传播


print(model)  # 查看网络结构
print(output.shape)  # 查看输出结果
