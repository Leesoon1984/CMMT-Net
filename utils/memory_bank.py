import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MemoryBank(nn.Module):
    def __init__(self, num_classes=4, queue_size=256, temperature=1, top_k=8):
        super(MemoryBank, self).__init__()

        self.num_classes = num_classes  # 类别数
        self.queue_size = queue_size  # 队列长度
        self.temperature = temperature  # 温度参数
        self.memory = [None] * num_classes
        self.flag = False
        self.top_k = top_k

    # 判断是否每一类都有特征了，有才开始计算对比损失
    def undate_flag(self):
        for i in range(self.num_classes):
            if self.memory[i] is None:
                self.flag = False
                return
        self.flag = True

    # 根据从预测准确的像素，根据置信度筛选
    def in_queue(self, confidence, label, contrast_feature):
        contrast_feature = contrast_feature.detach()
        label = label.detach().cpu()

        for c in range(self.num_classes):
            c_label = label == c
            c_contrast_feature = contrast_feature[c_label, :]
            c_confidence = confidence[c_label]

            if c_contrast_feature.shape[0] > 0:
                if c_contrast_feature.shape[0] <= self.top_k:
                    new_features = c_contrast_feature
                else:
                    values, indices = torch.topk(c_confidence, k=self.top_k)
                    new_features = c_contrast_feature[indices[0], :].unsqueeze(0)
                    for i in range(1, len(indices)):
                        new_features = torch.cat([new_features, c_contrast_feature[indices[i], :].unsqueeze(0)], dim=0)

                if self.memory[c] is None:
                    self.memory[c] = new_features
                else:
                    self.memory[c] = torch.cat([new_features, self.memory[c]], dim=0)[:self.queue_size, :]

        if not self.flag:
            self.undate_flag()

    def in_queue_get_loss(self, confidence, label, contrast_feature):
        contrast_feature = contrast_feature
        label = label.detach().cpu()
        loss = 0
        c_count = 0

        for c in range(self.num_classes):
            c_label = label == c
            c_contrast_feature = contrast_feature[c_label, :]
            c_confidence = confidence[c_label]

            if c_contrast_feature.shape[0] > 0:
                c_count += 1
                loss += self.c_contrast_loss(c, c_contrast_feature)

                if c_contrast_feature.shape[0] <= self.top_k:
                    new_features = c_contrast_feature
                else:
                    values, indices = torch.topk(c_confidence, k=self.top_k)
                    new_features = c_contrast_feature[indices[0], :].unsqueeze(0)
                    for i in range(1, len(indices)):
                        new_features = torch.cat([new_features, c_contrast_feature[indices[i], :].unsqueeze(0)], dim=0)

                if self.memory[c] is None:
                    self.memory[c] = new_features.detach()
                else:
                    self.memory[c] = torch.cat([new_features.detach(), self.memory[c]], dim=0)[:self.queue_size, :]

        if not self.flag:
            self.undate_flag()

        if c_count == 0:
            return 0
        else:
            return loss / c_count

    # 判断无标签像素是否预测准确，返回值shape为（B，H，W）
    def get_pseudo_label(self, contrast_feature):
        if not self.flag:
            return torch.full(contrast_feature.shape[:-1], 255, dtype=torch.uint8).cuda()
        dot_products = torch.zeros((self.num_classes,) + contrast_feature.shape[:-1])
        for c in range(self.num_classes):
            c_dot = torch.matmul(contrast_feature, self.memory[c].permute(1, 0)).mean(dim=-1)
            dot_products[c] = c_dot

        pseudo_label = torch.argmax(dot_products, dim=0).cuda()

        return pseudo_label

    # 对比学习损失
    def contrast_loss(self):
        loss = 0
        if not self.flag:
            return 0
        for i in range(self.num_classes):
            similarity_positive = torch.matmul(self.memory[i].cuda(), self.memory[i].permute(1, 0).cuda()).mean(dim=1)
            numerator = torch.exp(similarity_positive / self.temperature)
            denominator = numerator.clone()
            for j in range(self.num_classes):
                if j != i:
                    similarity_negative = torch.matmul(self.memory[i].cuda(), self.memory[j].permute(1, 0).cuda()).mean(dim=1)
                    similarity_negative = torch.exp(similarity_negative / self.temperature)
                    denominator += similarity_negative

            loss += -torch.log(numerator / denominator).mean()

        return loss / self.num_classes

    def c_contrast_loss(self, c, c_feature):
        if not self.flag:
            return 0

        similarity_positive = torch.matmul(c_feature, self.memory[c].permute(1, 0).cuda()).mean(dim=1)
        numerator = torch.exp(similarity_positive / self.temperature)
        denominator = numerator.clone()
        for i in range(self.num_classes):
            if i != c:
                similarity_negative = torch.matmul(c_feature, self.memory[i].permute(1, 0).cuda()).mean(dim=1)
                similarity_negative = torch.exp(similarity_negative / self.temperature)
                denominator += similarity_negative

        loss = -torch.log(numerator / denominator).mean()

        return loss
