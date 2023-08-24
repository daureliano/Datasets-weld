# -- coding: utf-8 --
# 计算模型的浮点计算量flops和参数量params
import torch
import torchvision
from thop import profile
# from timm.models.squeezenet01 import squeezenet1_0  # 第3个测试模型， BN 深度可分离卷积 残差
# from timm.models.resnet import resnet50
# from timm.models.mobilenetv3 import mobilenetv3_large_100
# from timm.models.inception_v4 import inception_v4
# from timm.models.eca_squeezenet02 import squeezenet1_0  # 20230503
from timm.models.squeezenet2 import squeezenet1_0  # 20230504

# model = resnet50()
# model = mobilenetv3_large_100()
# model = inception_v4()
model = squeezenet1_0()
input = torch.rand(1, 3, 224, 224)
flops, params = profile(model, (input,))
print('flops:', flops, 'params:', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
