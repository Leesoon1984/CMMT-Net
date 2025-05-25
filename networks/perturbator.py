import math, time
import torch
import torch.nn.functional as F
from torch import nn
import random
import numpy as np
import cv2
from torch.distributions.uniform import Uniform
import torchvision.transforms as transforms


class DropOut(nn.Module):
    def __init__(self, drop_rate=0.1, spatial_dropout=True):
        super(DropOut, self).__init__()
        self.dropout = nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.dropout(x)
        return x


class FeatureDrop(nn.Module):
    def __init__(self):
        super(FeatureDrop, self).__init__()

    def feature_dropout(self, x):
        """
            feature_dropout函数的作用是根据特征图的注意力（attention）来随机失活一些不重要的特征。
            它首先对每个通道求平均值，得到一个注意力图。然后对每个样本求最大值，得到一个阈值。
            阈值乘以一个随机因子，介于0.7和0.9之间。然后根据阈值生成一个掩码（mask），将注意力低于阈值的部分置为0，高于阈值的部分置为1。
            最后将掩码乘以原始特征图，得到失活后的特征图。
        """
        attention = torch.mean(x, dim=1, keepdim=True)  # 对每个通道求平均值，得到一个注意力图
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)  #
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def forward(self, x):
        x = self.feature_dropout(x)
        return x


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x


class Cutout(nn.Module):
    def __init__(self, size=8, p=0.5):
        super(Cutout, self).__init__()
        self.size = size
        self.p = p

    def forward(self, input):
        if np.random.rand() > self.p:
            return input

        h, w = input.shape[2], input.shape[3]
        mask = torch.ones((1, 1, h, w), dtype=torch.float32)

        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.size // 2, 0, h)
        y2 = np.clip(y + self.size // 2, 0, h)
        x1 = np.clip(x - self.size // 2, 0, w)
        x2 = np.clip(x + self.size // 2, 0, w)

        mask[:, :, y1:y2, x1:x2] = 0
        mask = mask.expand_as(input).cuda()

        input = input * mask
        return input


class Perturbator(nn.Module):
    def __init__(self):
        super(Perturbator, self).__init__()
        self.perturbator_list = []

        self.perturbator_list.append(DropOut())
        self.perturbator_list.append(FeatureDrop())
        self.perturbator_list.append(FeatureNoise())
        self.perturbator_list.append(Cutout())

    def forward(self, x, perturbator_index):
        x = self.perturbator_list[perturbator_index](x)
        return x




