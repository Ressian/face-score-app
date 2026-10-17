# -*- coding: utf-8 -*-
"""与生产权重结构完全一致的 ResNetFusion 定义。"""
import math
from collections import OrderedDict

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        m = OrderedDict()
        m["conv1"] = conv3x3(inplanes, planes, stride)
        m["bn1"] = nn.BatchNorm2d(planes)
        m["relu1"] = nn.ReLU(inplace=True)
        m["conv2"] = conv3x3(planes, planes)
        m["bn2"] = nn.BatchNorm2d(planes)
        self.group1 = nn.Sequential(m)
        self.relu = nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        residual = self.downsample(x) if self.downsample is not None else x
        out = self.relu(self.group1(x) + residual)
        return out


class ResNetFusion(nn.Module):
    def __init__(self, block, layers, geo_feat_dim=14):
        super().__init__()
        self.inplanes = 64
        m = OrderedDict()
        m["conv1"] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m["bn1"] = nn.BatchNorm2d(64)
        m["relu1"] = nn.ReLU(inplace=True)
        m["maxpool"] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.group1 = nn.Sequential(m)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.geo_mlp = nn.Sequential(
            nn.Linear(geo_feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
        )
        self.img_proj = nn.Linear(512 * block.expansion, 128)
        self.fusion_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.dropout = nn.Dropout(0.2)
        self.score_head = nn.Linear(128, 1)
        self.conf_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight.data)
                nn.init.constant_(module.bias.data, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, img, geo):
        x = self.group1(img)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        img_feat = self.img_proj(x)
        geo_feat = self.geo_mlp(geo)
        fused = self.fusion_head(torch.cat([img_feat, geo_feat], dim=-1))
        fused = self.dropout(fused)
        return self.score_head(fused), self.conf_head(fused)


def build_model(geo_dim=14):
    return ResNetFusion(BasicBlock, [2, 2, 2, 2], geo_feat_dim=geo_dim)

