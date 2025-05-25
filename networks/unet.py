# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import torch
import torch.nn as nn
from networks.perturbator import Perturbator
import random

class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""
    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)

class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, mode_upsampling=1):
        super(UpBlock, self).__init__()
        self.mode_upsampling = mode_upsampling
        if mode_upsampling==0:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        elif mode_upsampling==1:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif mode_upsampling==2:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        elif mode_upsampling==3:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.mode_upsampling != 0:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, mode_upsampling=self.up_type)

        # self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=1, padding=0, bias=True)
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature, is_feat=False):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        if not is_feat:
            return output
        else:
            return output, x

    
class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)

    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        return output1

class MCNet2d_v1(nn.Module):
    def __init__(self, in_chns, class_num):
        super(MCNet2d_v1, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 0,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)
        
    def forward(self, x, is_feat=False):
        feature = self.encoder(x)
        if not is_feat:
            output1 = self.decoder1(feature)
            output2 = self.decoder2(feature)
            return output1, output2
        else:
            output1, feat1 = self.decoder1(feature, is_feat=is_feat)
            output2, feat2 = self.decoder2(feature, is_feat=is_feat)
            return [output1, output2], [feat1, feat2]

class MCNet2d_v2(nn.Module):
    def __init__(self, in_chns, class_num):
        super(MCNet2d_v2, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 0,
                  'acti_func': 'relu'}
        params3 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 2,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)
        self.decoder3 = Decoder(params3)
        
    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        output2 = self.decoder2(feature)
        output3 = self.decoder3(feature)
        return output1, output2, output3

class MCNet2d_v3(nn.Module):
    def __init__(self, in_chns, class_num):
        super(MCNet2d_v3, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 0,
                  'acti_func': 'relu'}
        params3 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 2,
                  'acti_func': 'relu'}
        params4 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 3,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)
        self.decoder3 = Decoder(params3)
        self.decoder4 = Decoder(params4)
        
    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        output2 = self.decoder2(feature)
        output3 = self.decoder3(feature)    
        output4 = self.decoder4(feature)
        return output1, output2, output3, output4

class Decoder_Dual(nn.Module):
    def __init__(self, args, params, use_MLP=False):
        super(Decoder_Dual, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']

        self.use_MLP = use_MLP
        self.use_dropout = args.use_dropout

        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, mode_upsampling=self.up_type)

        # self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=1, padding=0, bias=True)
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

        if self.use_MLP:
            # max the discrepancy between the output feature after concatenation of c1 and c4
            if args.use_norm:
                self.mapping = nn.Sequential(nn.Conv2d(self.ft_chns[0], self.ft_chns[0], 1, bias=False), nn.BatchNorm2d(self.ft_chns[0]), nn.ReLU(True))
                if args.use_dropout:
                    self.dropout = nn.Dropout2d(p=args.dropout)
            else:
                self.mapping = nn.Conv2d(self.ft_chns[0], self.ft_chns[0], 1, bias=False)


    def forward(self, feature, use_MLP=False):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

    
        if self.use_MLP:
            x = self.mapping(x)
            if self.use_dropout:
                x = self.dropout(x)
            # return_feature = x
        # else:
        #     return_feature = feature

        output = self.out_conv(x)

        return output, x



########### 将mapping layer加入较浅层：即Encoder的最后一层
class Decoder_Dual_v3(nn.Module):
    def __init__(self, args, params, use_MLP=False):
        super(Decoder_Dual_v3, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']

        self.use_MLP = use_MLP
        self.use_dropout = args.use_dropout

        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, mode_upsampling=self.up_type)

        # self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=1, padding=0, bias=True)
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

        if self.use_MLP:
            # max the discrepancy between the output feature after concatenation of c1 and c4
            if args.use_norm:
                self.mapping = nn.Sequential(nn.Conv2d(self.ft_chns[4], self.ft_chns[4], 1, bias=False), nn.BatchNorm2d(self.ft_chns[4]), nn.ReLU(True))
                if args.use_dropout:
                    self.dropout = nn.Dropout2d(p=args.dropout)
            else:
                self.mapping = nn.Conv2d(self.ft_chns[4], self.ft_chns[4], 1, bias=False)


    def forward(self, feature, use_MLP=False):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]


        if self.use_MLP:
            x4 = self.mapping(x4)
            if self.use_dropout:
                x4 = self.dropout(x4)
            # return_feature = x
        # else:
        #     return_feature = feature

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

    
        output = self.out_conv(x)

        return output, x4


########### 将mapping layer加入较浅层：即Encoder的最后一层,同时返回decoder最后一层特征
class Decoder_Dual_v4(nn.Module):
    def __init__(self, args, params, use_MLP=False):
        super(Decoder_Dual_v4, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']

        self.use_MLP = use_MLP
        self.use_dropout = args.use_dropout

        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, mode_upsampling=self.up_type)

        # self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=1, padding=0, bias=True)
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

        if self.use_MLP:
            # max the discrepancy between the output feature after concatenation of c1 and c4
            if args.use_norm:
                self.mapping = nn.Sequential(nn.Conv2d(self.ft_chns[4], self.ft_chns[4], 1, bias=False), nn.BatchNorm2d(self.ft_chns[4]), nn.ReLU(True))
                if args.use_dropout:
                    self.dropout = nn.Dropout2d(p=args.dropout)
            else:
                self.mapping = nn.Conv2d(self.ft_chns[4], self.ft_chns[4], 1, bias=False)


    def forward(self, feature, use_MLP=False):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]


        if self.use_MLP:
            x4 = self.mapping(x4)
            if self.use_dropout:
                x4 = self.dropout(x4)
            # return_feature = x
        # else:
        #     return_feature = feature

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

    
        output = self.out_conv(x)

        return output, x4, x
    


class MCNet2d_Dual_v1(nn.Module):
    def __init__(self, args, params, in_chns, class_num, use_MLP=False):
        super(MCNet2d_Dual_v1, self).__init__()

        self.encoder = Encoder(params)

        self.decoder = Decoder_Dual(args, params, use_MLP=use_MLP)

        
    def forward(self, x):

        feature = self.encoder(x)
        output, feat = self.decoder(x)

        return output, feat


class MCNet2d_Dual_v2(nn.Module):
    def __init__(self, args, params, in_chns, class_num, use_MLP=False):
        super(MCNet2d_Dual_v2, self).__init__()

        # self.encoder = Encoder(params)

        self.decoder = Decoder_Dual(args, params, use_MLP=use_MLP)

        
    def forward(self, x):

        # feature = self.encoder(x)
        output, feat = self.decoder(x)

        return output, feat


class MCNet2d_Dual_DualPlus_v2(nn.Module):
    def __init__(self, args, in_chns, class_num):
        super(MCNet2d_Dual_DualPlus_v2, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 0,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params1)

        # three branch, branch1 is the main branch without modification, the other two branches can add mapping layer
        self.branch1 = Decoder_Dual(args, params=params1, use_MLP=args.use_MLP)

        if args.mode_mapping == 'both':
            self.branch2 = Decoder_Dual(args, params=params2,  use_MLP=args.use_MLP)
        else:
            self.branch2 = Decoder_Dual(args, params=params2)

    def forward(self, x):
        # logits = {}

        feature = self.encoder(x)
      
        pred1, feature1 = self.branch1(feature)
        pred2, feature2 = self.branch2(feature)
        
        # logits['pred1'] = pred1
        # logits['feature1'] = feature1
        # logits['pred2'] = pred2
        # logits['feature2'] = feature2
        
        # return [output1, output2], [feat1, feat2]

        return [pred1, pred2], [feature1, feature2]



class MCNet2d_Dual_v2(nn.Module):
    def __init__(self, args, params, in_chns, class_num, use_MLP=False):
        super(MCNet2d_Dual_v2, self).__init__()

        # self.encoder = Encoder(params)

        self.decoder = Decoder_Dual(args, params, use_MLP=use_MLP)

        
    def forward(self, x):

        # feature = self.encoder(x)
        output, feat = self.decoder(x)

        return output, feat


class MCNet2d_Dual_DualPlus_v3(nn.Module):
    def __init__(self, args, in_chns, class_num):
        super(MCNet2d_Dual_DualPlus_v3, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 0,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params1)

        # three branch, branch1 is the main branch without modification, the other two branches can add mapping layer
        self.branch1 = Decoder_Dual_v3(args, params=params1, use_MLP=args.use_MLP)

        if args.mode_mapping == 'both':
            self.branch2 = Decoder_Dual_v3(args, params=params2,  use_MLP=args.use_MLP)
        else:
            self.branch2 = Decoder_Dual_v3(args, params=params2)

    def forward(self, x):
        # logits = {}

        feature = self.encoder(x)
      
        pred1, feature1 = self.branch1(feature)
        pred2, feature2 = self.branch2(feature)
        
        # logits['pred1'] = pred1
        # logits['feature1'] = feature1
        # logits['pred2'] = pred2
        # logits['feature2'] = feature2
        
        # return [output1, output2], [feat1, feat2]

        return [pred1, pred2], [feature1, feature2]



### 加入特征级别的抖动以及对比学习方法
class MCNet2d_Dual_DualPlus_v4(nn.Module):
    def __init__(self, args, in_chns, class_num, feature_length):
        super(MCNet2d_Dual_DualPlus_v4, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 0,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params1)

        # three branch, branch1 is the main branch without modification, the other two branches can add mapping layer
        self.branch1 = Decoder_Dual_v4(args, params=params1, use_MLP=args.use_MLP)

        if args.mode_mapping == 'both':
            self.branch2 = Decoder_Dual_v4(args, params=params2,  use_MLP=args.use_MLP)
        else:
            self.branch2 = Decoder_Dual_v4(args, params=params2)
        self.perturbator = Perturbator()
        self.projection_head = ProjectionHead(dim_in=16, dim_out=feature_length)

    def forward(self, x, mode='train'):
        # logits = {}

        feature1 = self.encoder(x)
        feature2 = feature1#[:-1]
        
      
        # if mode == 'train':
            # f4 = feature1[-1].clone()
            # index1, = random.sample(range(0, len(self.perturbator.perturbator_list)), 1)
            # feature2.append(self.perturbator(f4, index1))
        pred1, feature1_, pre_project1 = self.branch1(feature1)
        pred2, feature2_, pre_project2 = self.branch2(feature2)

        return [pred1, pred2], [feature1, feature2], [pre_project1, pre_project2]
        # else:
            # pred1, feature1, pre_project1 = self.branch1(feature1)
            # return [pred1, pred2], [pre_project1, pre_project2]


    def forward_projection_head(self, features):
        return self.projection_head(features)
    

class MCNet2d_DualPlus(nn.Module):
    def __init__(self, args, in_chns, class_num):
        super(MCNet2d_DualPlus, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 0,
                  'acti_func': 'relu'}

        # three branch, branch1 is the main branch without modification, the other two branches can add mapping layer
        self.branch1 = MCNet2d_Dual_v1(args, params=params2, in_chns=in_chns, class_num=class_num, use_MLP=args.use_MLP)


        if args.mode_mapping == 'both':
            self.branch2 = MCNet2d_Dual_v1(args, params=params1, in_chns=in_chns, class_num=class_num, use_MLP=args.use_MLP)
        else:
            self.branch2 = MCNet2d_Dual_v1(args, params=params1, in_chns=in_chns, class_num=class_num)

        self.branch2 = MCNet2d_Dual(args, params=params2, in_chns=in_chns, class_num=class_num, use_MLP=args.use_MLP)

    def forward(self, x):
        # logits = {}

      
        pred1, feature1 = self.branch1(x)
        pred2, feature2 = self.branch2(x)
        
        # logits['pred1'] = pred1
        # logits['feature1'] = feature1
        # logits['pred2'] = pred2
        # logits['feature2'] = feature2
        
        # return [output1, output2], [feat1, feat2]

        return [pred1, pred2], [feature1, feature2]


class ProjectionHead(nn.Module):
    def __init__(self, dim_in=16, dim_out=256):
        super(ProjectionHead, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.BatchNorm1d(dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim_out)
        )

    def forward(self, x):
        # x的batch为1时，复制一个一样的，否则batchnorm会报错
        ori_batch = x.shape[0]
        if ori_batch == 1:
            x = x.repeat(2, 1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1)   # 对特征进行归一化操作
        if ori_batch == 1:
            return x[0].unsqueeze(0)
        return x



if __name__ == '__main__':

    import argparse
    # compute FLOPS & PARAMETERS7

    parser = argparse.ArgumentParser()

    parser.add_argument('--use_MLP', default=True, type=bool)
    parser.add_argument('--use_norm', default=True, type=bool)
    parser.add_argument('--use_dropout', default=True, type=bool)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--mode_mapping', default='else', type=str)                         # both or else, the only difference is whether to use mapping on branch1
    args = parser.parse_args()


    from ptflops import get_model_complexity_info

    # model = UNet(in_chns=1, class_num=4).cuda()
    model = MCNet2d_DualPlus(args, in_chns=1, class_num=4).cuda()

    with torch.cuda.device(0):
      macs, params = get_model_complexity_info(model, (1, 256, 256), as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
      print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
      print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    import ipdb; ipdb.set_trace()
