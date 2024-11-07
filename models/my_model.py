import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small


#输入数据预处理，如果输入的四元数旋转矩阵未提供，则填充默认值；如果输入的深度图像尺寸不正确，则调整其尺寸。
def refine_inputs(X):

    # fill quaternion rotation if not given
    # make it [1, 0, 0, 0] repeated with numrows = X[0].shape[0]
    if X[2] is None:
        # X[2] = torch.Tensor([1, 0, 0, 0]).float()
        X[2] = torch.zeros((X[0].shape[0], 4)).float().to(X[0].device)
        X[2][:, 0] = 1

    # if input depth images are not of right shape, resize
    if X[0].shape[-2] != 60 or X[0].shape[-1] != 90:
        X[0] = F.interpolate(X[0], size=(60, 90), mode='bilinear')

    return X

class MobileNet(nn.Module):
    def __init__(self, output_dim: int): 
        super(MobileNet, self).__init__()
        self.cnn = mobilenet_v3_small(pretrained=False)
        self.cnn.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.cnn.classifier = nn.Linear(576, output_dim)
        self.features_dim = output_dim

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.cnn(depth)