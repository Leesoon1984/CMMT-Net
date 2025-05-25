import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling_function(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', mode_upsampling = 1):
        super(Upsampling_function, self).__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res
    
    
class Decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, up_type=0):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization, mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization, mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization, mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization, mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]
        
        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)
        
        return out_seg
 
class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(VNet, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
    
    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        return out_seg1

class MCNet3d_v1(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(MCNet3d_v1, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters,normalization,  has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
    
    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        out_seg2 = self.decoder2(features)
        return out_seg1, out_seg2
    
class CCNet3d_v1(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(CCNet3d_v1, self).__init__()

        self.encoder1 = Encoder(n_channels, n_classes, n_filters,normalization,  has_dropout, has_residual)
        self.encoder2 = Encoder(n_channels, n_classes, n_filters,normalization,  has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)

    def forward(self, input):
        features1 = self.encoder1(input)
        features2 = self.encoder2(input)
        out_seg1 = self.decoder1(features1)
        out_seg2 = self.decoder2(features2)
        return out_seg1, out_seg2
    
class MCNet3d_v2(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(MCNet3d_v2, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        self.decoder3 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 2)
    
    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        out_seg2 = self.decoder2(features)
        out_seg3 = self.decoder3(features)
        return out_seg1, out_seg2, out_seg3


class center_model(nn.Module):

    def __init__(self, num_classes, ndf=64, out_channel=1):
        super(center_model, self).__init__()
        # downsample 16
        self.conv0 = nn.Conv3d(1, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.avgpool = nn.AvgPool3d((7, 7, 5))
        # self.avgpool = nn.AvgPool3d((5, 7, 7))
        # self.avgpool = nn.AvgPool3d((5, 16, 16))
        self.fc1 = nn.Linear(ndf * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout3d(0.5)
        self.Softmax = nn.Softmax()
        self.out = nn.Conv3d(ndf * 2, num_classes, kernel_size=1)

    def forward(self, map):
        batch_size = map.shape[0]  # patch_size = (112, 112, 80)
        map_feature = self.conv0(map)  # (2,112,112,80)->(64,56,56,40)
        x = self.leaky_relu(map_feature)
        x = self.dropout(x)

        x = self.conv1(x)  # (64,56,56,40)->(128,28,28,20)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        # x = self.out(x)

        x = self.conv2(x)  # (128,28,28,20)->(256,14,14,10)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)  # (256,14,14,10)->(512,7,7,5)
        x = self.leaky_relu(x)

        x = self.avgpool(x)  # (512)

        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class Decoder_Dual(nn.Module):
    def __init__(self, args, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, use_MLP=False, up_type=0):
        super(Decoder_Dual, self).__init__()
        self.has_dropout = has_dropout

        self.use_MLP = use_MLP
        self.use_dropout = args.use_dropout  ## used for MLP Layer

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization, mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization, mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization, mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization, mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        if self.use_MLP:
            # max the discrepancy between the output feature after concatenation of c1 and c4
            if args.use_norm:
                self.mapping = nn.Sequential(nn.Conv3d( n_filters, n_filters, 1, bias=False), nn.BatchNorm3d(n_filters), nn.ReLU(True))
                if args.use_dropout:
                    self.dropout = nn.Dropout3d(p=args.dropout, inplace=False)
            else:
                self.mapping = nn.Conv3d( n_filters, n_filters, 1, bias=False)


    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]
        
        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)

        if self.use_MLP:
           x9  = self.mapping(x9)
           if self.use_dropout:
                x9 = self.dropout(x9)
        elif self.has_dropout:
            x9 = self.dropout(x9)

        # if self.has_dropout:
        #     x9 = self.dropout(x9)

        # elif self.use_MLP:
        #    x9  = self.mapping(x9)
        #    if self.use_dropout:
        #         x9 = self.dropout(x9)

        out_seg = self.out_conv(x9)
        
        return out_seg, x9
    

class Decoder_Dual_v4(nn.Module):
    def __init__(self, args, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, use_MLP=False, up_type=0):
        super(Decoder_Dual_v4, self).__init__()
        self.has_dropout = has_dropout

        self.use_MLP = use_MLP
        self.use_dropout = args.use_dropout  ## used for MLP Layer

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization, mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization, mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization, mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization, mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        if self.use_MLP:
            # max the discrepancy between the output feature after concatenation of c1 and c4
            if args.use_norm:
                self.mapping = nn.Sequential(nn.Conv3d( n_filters * 16, n_filters * 16, 1, bias=False), nn.BatchNorm3d(n_filters * 16), nn.ReLU(True))
                if args.use_dropout:
                    self.dropout = nn.Dropout3d(p=args.dropout, inplace=False)
            else:
                self.mapping = nn.Conv3d( n_filters, n_filters, 1, bias=False)


    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]
        

        if self.use_MLP:
           x5  = self.mapping(x5)
           if self.use_dropout:
                x5 = self.dropout(x5)
        elif self.has_dropout:
            x5 = self.dropout(x5)



        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)


        # if self.has_dropout:
        #     x9 = self.dropout(x9)

        # elif self.use_MLP:
        #    x9  = self.mapping(x9)
        #    if self.use_dropout:
        #         x9 = self.dropout(x9)

        out_seg = self.out_conv(x9)
        
        return out_seg, x9
    

class MCNet3d_Dual(nn.Module):
    def __init__(self, args, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(MCNet3d_Dual, self).__init__()


        self.encoder = Encoder(n_channels, n_classes, n_filters,normalization,  has_dropout, has_residual)

        if args.mode_mapping == 'both':
            self.decoder1 = Decoder_Dual(args, n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, args.use_MLP, 0)
        else:
            self.decoder1 = Decoder_Dual(args, n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder_Dual(args, n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, args.use_MLP, 1)  ### 使用MLP层
    
    def forward(self, input):
        features = self.encoder(input)
        out_seg1, feat1 = self.decoder1(features)
        out_seg2, feat2 = self.decoder2(features)
        return [out_seg1, out_seg2], [feat1, feat2]



class MCNet3d_Dual_v4(nn.Module):
    def __init__(self, args, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(MCNet3d_Dual_v4, self).__init__()


        self.encoder = Encoder(n_channels, n_classes, n_filters,normalization,  has_dropout, has_residual)

        if args.mode_mapping == 'both':
            self.decoder1 = Decoder_Dual_v4(args, n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, args.use_MLP, 0)
        else:
            self.decoder1 = Decoder_Dual_v4(args, n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder_Dual(args, n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, args.use_MLP, 1)  ### 使用MLP层
    
    def forward(self, input):
        features = self.encoder(input)
        out_seg1, feat1 = self.decoder1(features)
        out_seg2, feat2 = self.decoder2(features)
        return [out_seg1, out_seg2], [feat1, feat2]


    
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

    # compute FLOPS & PARAMETERS
    from ptflops import get_model_complexity_info
    # model = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=False)
    model = MCNet3d__Dual(args, n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=False)
    with torch.cuda.device(0):
      macs, params = get_model_complexity_info(model, (1, 112, 112, 80), as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
      print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
      print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    with torch.cuda.device(0):
      macs, params = get_model_complexity_info(model, (1, 96, 96, 96), as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
      print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
      print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # import ipdb; ipdb.set_trace()
