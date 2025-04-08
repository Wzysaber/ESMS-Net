import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init


from model.STranU.Resnet_part import ResidualNet
from model.STranU.Swin_transformer_part import SwinTrans_x_downsample


class Decoder_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample_4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)
        )
        self.stage_up_4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.upsample_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        )
        self.stage_up_3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.upsample_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        )
        self.stage_up_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.upsample_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=48, kernel_size=4, stride=2, padding=1)
        )
        self.stage_up_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conT1 = nn.ConvTranspose2d(48, 16, kernel_size=2, stride=2)

    def forward(self, x):
        DT_UnetX1 = x[0]
        DT_UnetX2 = x[1]
        DT_UnetX3 = x[2]
        DT_UnetX4 = x[3]

        up_4_conv = self.stage_up_4(DT_UnetX4)
        DT_UnetX4_up = self.upsample_4(up_4_conv)

        up_3_conv = torch.cat([DT_UnetX4_up, DT_UnetX3], dim=1)
        up_3_conv = self.stage_up_3(up_3_conv)
        DT_UnetX3_up = self.upsample_3(up_3_conv)

        up_2_conv = torch.cat([DT_UnetX3_up, DT_UnetX2], dim=1)
        up_2_conv = self.stage_up_2(up_2_conv)
        DT_UnetX2_up = self.upsample_2(up_2_conv)

        up_1_conv = torch.cat([DT_UnetX2_up, DT_UnetX1], dim=1)
        up_1_conv = self.stage_up_1(up_1_conv)
        DT_UnetX1_up = self.upsample_1(up_1_conv)

        x = self.conT1(DT_UnetX1_up)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class DT_Unet_Encode(nn.Module):
    def __init__(self):
        super(DT_Unet_Encode, self).__init__()

        self.ResidualNet = ResidualNet(depth=50)
        self.Swin_transform_x_downsample = SwinTrans_x_downsample()


        inchannel_Swin_transform = [96, 192, 384, 768]
        outchannel = [128, 256, 512, 1024]

        self.Conv3 = nn.ModuleList()

        for i in range(4):
            Conv3x3 = BasicConv(in_planes=inchannel_Swin_transform[i], out_planes=outchannel[i], kernel_size=1,
                                stride=1,
                                groups=1, relu=True, bn=True, bias=False)

            self.Conv3.append(Conv3x3)

    def forward(self, x):
        Resnet_Encode = self.ResidualNet(x)
        Swin_transform_Encode = self.Swin_transform_x_downsample(x)

        DT_Unet_Encode = []
        for i in range(4):
            S1 = Swin_transform_Encode[i]
            S1 = self.Conv3[i](S1)

            A1 = Resnet_Encode[i]

            x = S1 + A1
            DT_Unet_Encode.append(x)

        return DT_Unet_Encode


class STranU(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.DT_Unet_Encode = DT_Unet_Encode()
        self.Decoder_block = Decoder_block()
        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=6,
            kernel_size=3,
        )

    def forward(self, x):
        x = self.DT_Unet_Encode(x)
        x = self.Decoder_block(x)
        logits = self.segmentation_head(x)
        return logits


if __name__ == '__main__':
    model = STranU()
    model.eval()
    image = torch.randn(32, 3, 256, 256)

    output = model(image)
    print("input:", image.shape)
    print("output:", output.shape)
    # print("output_stage0:", output[0].shape)
    # print("output_stage1:", output[1].shape)
    # print("output_stage2:", output[2].shape)
    # print("output_stage3:", output[3].shape)
