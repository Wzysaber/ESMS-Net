import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial


class SoftPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=1):
        super(SoftPool2d, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, stride, padding)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features, eps=1e-5),
        )

        self.proj = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.proj_act = nn.GELU()
        self.proj_bn = nn.BatchNorm2d(hidden_features, eps=1e-5)

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_features, eps=1e-5),
        )

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.drop(x)

        x = self.proj(x) + x
        x = self.proj_act(x)
        x = self.proj_bn(x)

        x = self.conv2(x)

        x = self.drop(x)
        x = x.permute(0, 2, 3, 1)
        return x


class Atrous_Attention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        # B, C//3, H, W
        B, d, H, W = q.shape
        q = q.reshape([B, d // self.head_dim, self.head_dim, 1, H * W]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d
        k = self.unfold(k).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 2,
                                                                                                        3)  # B,h,N,d,k*k
        attn = (q @ k) * self.scale  # B,h,N,1,k*k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 3,
                                                                                                        2)  # B,h,N,k*k,d
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x


class WindowAtrousAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [Atrous_Attention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)
        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        x = x.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 3, 4, 2).clone()

        for i in range(self.num_dilation):
            x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])
        x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Trans_conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.ASPP_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.trans_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)

        )

    def forward(self, x, Trans_x1, Trans_x2, Trans_x3):
        H = x.size()[2]  # (== h/16)
        W = x.size()[3]
        x_con1x1 = self.Conv(x)

        Trans_x1 = Trans_x1.permute(0, 3, 1, 2)
        Trans_x2 = Trans_x2.permute(0, 3, 1, 2)
        Trans_x3 = Trans_x3.permute(0, 3, 1, 2)

        x_pool = self.ASPP_pooling(x)
        x_pool = F.interpolate(x_pool, size=(H, W), mode="bilinear",
                               align_corners=False)  # (shape: (batch_size, 256, h/16, w/16))

        Trans_x = torch.cat([x_con1x1, Trans_x1, Trans_x2, Trans_x3, x_pool], dim=1)
        Trans_x = self.trans_conv(Trans_x)
        Trans_x = Trans_x.permute(0, 2, 3, 1)

        return Trans_x


class PPAFormer(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, kernel_size=3, dilation=[1, 2],
                 drop=0., mlp_ratio=4., act_layer=nn.GELU,
                 Close_attn=None
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.Trans_conv = Trans_conv(dim * 5, dim)

        self.attn1 = WindowAtrousAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                               qk_scale=qk_scale,
                                               attn_drop=attn_drop, kernel_size=kernel_size, dilation=[1])
        self.attn2 = WindowAtrousAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                               qk_scale=qk_scale,
                                               attn_drop=attn_drop, kernel_size=kernel_size, dilation=[2])
        self.attn3 = WindowAtrousAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                               qk_scale=qk_scale,
                                               attn_drop=attn_drop, kernel_size=kernel_size, dilation=[3])

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        Transx = x.permute(0, 2, 3, 1)

        Trans_x1 = Transx + self.drop_path(self.attn1(self.norm1(Transx)))
        Trans_x2 = Transx + self.drop_path(self.attn2(self.norm1(Transx)))
        Trans_x3 = Transx + self.drop_path(self.attn3(self.norm1(Transx)))

        Trans_x = self.Trans_conv(x, Trans_x1, Trans_x2, Trans_x3)

        Trans_x = Trans_x + self.drop_path(self.mlp(self.norm2(Trans_x)))
        Trans_x = Trans_x.permute(0, 3, 1, 2)

        return Trans_x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out


class StdConv2d(nn.Conv2d):
    # StdConv2d(3, 64, kernel_size=7, stride=2, bias=False, padding=3))
    def forward(self, x):
        w = self.weight  # [64, 3, 7, 7] 64 ge channel 3 7x7
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        # x = x.transpose([0, 3, 1, 2])
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return StdConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = StdConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(32, planes, eps=1e-6)

        self.conv2 = StdConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.GroupNorm(32, planes, eps=1e-6)

        self.conv3 = StdConv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(32, planes * 4, eps=1e-6)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

    # depths=[3, 4, 6, 3]


class WAA_attn(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 drop_path=0.5, norm_layer=nn.LayerNorm, kernel_size=3,
                 drop=0., mlp_ratio=4., act_layer=nn.GELU,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.attn1 = WindowAtrousAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                               qk_scale=qk_scale,
                                               attn_drop=attn_drop, kernel_size=kernel_size, dilation=[1, 2, 3, 4])

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        Transx = x.permute(0, 2, 3, 1)
        Trans_x = Transx + self.drop_path(self.attn1(self.norm1(Transx)))
        Trans_x = Trans_x + self.drop_path(self.mlp(self.norm2(Trans_x)))
        Trans_x = Trans_x.permute(0, 3, 1, 2)

        return Trans_x


class MSFA(nn.Module):
    def __init__(self, in_channel=32):
        super(MSFA, self).__init__()
        out_channel = in_channel
        self.conv_1x1_1 = StdConv2d(in_channel, out_channel, kernel_size=1)
        self.bn_conv_1x1_1 = nn.GroupNorm(32, out_channel, eps=1e-6)

        self.conv_3x3_1 = StdConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=3, dilation=3)  # 6
        self.bn_conv_3x3_1 = nn.GroupNorm(32, out_channel, eps=1e-6)

        self.conv_3x3_2 = StdConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=6, dilation=6)  # 12
        self.bn_conv_3x3_2 = nn.GroupNorm(32, out_channel, eps=1e-6)

        self.conv_3x3_3 = StdConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=8, dilation=8)  # 18
        self.bn_conv_3x3_3 = nn.GroupNorm(32, out_channel, eps=1e-6)

        self.conv_cat = nn.Sequential(
            StdConv2d(in_channel * 4, in_channel, 1, 1, padding=0, bias=True),
            nn.GroupNorm(32, out_channel, eps=1e-6),
            nn.ReLU(inplace=True),
        )

        self.ASPP_attn = WAA_attn(in_channel)

    def forward(self, x):
        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x)))  # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(x)))  # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(x)))  # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(x)))  # (shape: (batch_size, 256, h/16, w/16))

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3], 1)
        out = self.conv_cat(out)

        out = self.ASPP_attn(out)

        return out


def image_makeing(x, pred):
    _, pred_labels = torch.max(pred, dim=1)

    matrix = pred_labels

    segmentation_mask = torch.ones_like(matrix)

    # 将标签中为 0 和 1 的位置置为 0
    segmentation_mask[(matrix == 0) | (matrix == 1)] = 0

    mask = segmentation_mask.unsqueeze(1)
    mask = mask.expand(-1, 3, -1, -1)

    image2 = mask * x

    return image2


class ResNet(nn.Module):
    def __init__(self, depths=[3, 4, 6, 3],
                 num_heads=[2, 4, 8, 16], dilation=[1, 2],
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 qkv_bias=True, qk_scale=None, attn_drop=0., drop_path=0.1,
                 drop=0., mlp_ratio=4., act_layer=nn.GELU,
                 Close_attn="MSDA"
                 ):
        self.inplanes = 32
        super(ResNet, self).__init__()

        # different model config between ImageNet and CIFAR

        self.conv1 = StdConv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]

        self.layer1 = self.stage_making(Bottleneck, PPAFormer,
                                        32, depths[0],
                                        num_heads=num_heads[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        attn_drop=attn_drop, dilation=dilation,
                                        drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],
                                        norm_layer=norm_layer,
                                        drop=drop, mlp_ratio=mlp_ratio, act_layer=act_layer,
                                        Close_attn="MSDA"
                                        )

        self.layer2 = self.stage_making(Bottleneck, PPAFormer,
                                        64, depths[1], stride=2,
                                        num_heads=num_heads[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        attn_drop=attn_drop, dilation=dilation,
                                        drop_path=dpr[sum(depths[:1]):sum(depths[:1 + 1])],
                                        norm_layer=norm_layer,
                                        drop=drop, mlp_ratio=mlp_ratio, act_layer=act_layer,
                                        Close_attn="MSDA"
                                        )

        self.layer3 = self.stage_making(Bottleneck, PPAFormer,
                                        128, depths[2], stride=2,
                                        num_heads=num_heads[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        attn_drop=attn_drop, dilation=dilation,
                                        drop_path=dpr[sum(depths[:2]):sum(depths[:2 + 1])],
                                        norm_layer=norm_layer,
                                        drop=drop, mlp_ratio=mlp_ratio, act_layer=act_layer,
                                        Close_attn="MSDA"
                                        )

        self.layer4 = self.stage_making(Bottleneck, PPAFormer,
                                        256, depths[3], stride=2,
                                        num_heads=num_heads[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        attn_drop=attn_drop, dilation=dilation,
                                        drop_path=dpr[sum(depths[:3]):sum(depths[:3 + 1])],
                                        norm_layer=norm_layer,
                                        drop=drop, mlp_ratio=mlp_ratio, act_layer=act_layer,
                                        Close_attn="MSDA"
                                        )

    def stage_making(self, block, PPAFormer,
                     planes, blocks, stride=1,
                     num_heads=2, qkv_bias=True, qk_scale=None,
                     attn_drop=0., drop_path=0., dilation=[1, 2],
                     norm_layer=partial(nn.LayerNorm, eps=1e-6),
                     drop=0., mlp_ratio=4., act_layer=nn.GELU,
                     Close_attn="MSDA"
                     ):

        downsample = None
        drop_path_data = []
        layers = []

        for i in range(blocks):
            data = drop_path[i] if isinstance(drop_path, list) else drop_path
            drop_path_data.append(data)

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                StdConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, planes * block.expansion, eps=1e-6),
            )

        layers.append(block(self.inplanes, planes, stride, downsample))
        layers.append(PPAFormer(
            dim=planes * block.expansion, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            drop_path=drop_path_data[0],
            norm_layer=norm_layer,
            kernel_size=3, dilation=dilation,
            drop=drop, mlp_ratio=mlp_ratio, act_layer=act_layer,
            Close_attn=Close_attn
        ))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            layers.append(PPAFormer(dim=self.inplanes, num_heads=num_heads,
                               qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                               drop_path=drop_path_data[i],
                               norm_layer=norm_layer,
                               kernel_size=3, dilation=dilation,
                               drop=drop, mlp_ratio=mlp_ratio, act_layer=act_layer,
                               Close_attn=Close_attn
                               ))

        return nn.Sequential(*layers)

    def forward(self, x):
        stageData = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        stageData.append(x)

        x = self.layer2(x)
        stageData.append(x)

        x = self.layer3(x)
        stageData.append(x)

        x = self.layer4(x)
        stageData.append(x)

        return stageData


class Auxiliary_Enhanced_Encoder(nn.Module):
    def __init__(self, depths=[3, 4, 6, 3],
                 num_heads=[2, 4, 8, 16], dilation=[1, 2],
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 qkv_bias=True, qk_scale=None, attn_drop=0., drop_path=0.1,
                 drop=0., mlp_ratio=4., act_layer=nn.GELU,
                 ):
        self.inplanes = 32
        super(Auxiliary_Enhanced_Encoder, self).__init__()

        # different model config between ImageNet and CIFAR

        self.conv1 = StdConv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]

        self.layer1 = self.stage_making(Bottleneck, PPAFormer,
                                        32, depths[0],
                                        num_heads=num_heads[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        attn_drop=attn_drop, dilation=dilation,
                                        drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],
                                        norm_layer=norm_layer,
                                        drop=drop, mlp_ratio=mlp_ratio, act_layer=act_layer,
                                        Close_attn="MSDA"
                                        )

        self.layer2 = self.stage_making(Bottleneck, PPAFormer,
                                        64, depths[1], stride=2,
                                        num_heads=num_heads[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        attn_drop=attn_drop, dilation=dilation,
                                        drop_path=dpr[sum(depths[:1]):sum(depths[:1 + 1])],
                                        norm_layer=norm_layer,
                                        drop=drop, mlp_ratio=mlp_ratio, act_layer=act_layer,
                                        Close_attn="MSDA"
                                        )

        self.ConvdilateAttn1 = MSFA(128)
        self.ConvdilateAttn2 = MSFA(256)

    def stage_making(self, block, PPAFormer,
                     planes, blocks, stride=1,
                     num_heads=2, qkv_bias=True, qk_scale=None,
                     attn_drop=0., drop_path=0., dilation=[1, 2],
                     norm_layer=partial(nn.LayerNorm, eps=1e-6),
                     drop=0., mlp_ratio=4., act_layer=nn.GELU,
                     Close_attn="MSDA"
                     ):

        downsample = None
        drop_path_data = []
        layers = []

        for i in range(blocks):
            data = drop_path[i] if isinstance(drop_path, list) else drop_path
            drop_path_data.append(data)

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                StdConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, planes * block.expansion, eps=1e-6),
            )

        layers.append(block(self.inplanes, planes, stride, downsample))
        layers.append(PPAFormer(
            dim=planes * block.expansion, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            drop_path=drop_path_data[0],
            norm_layer=norm_layer,
            kernel_size=3, dilation=dilation,
            drop=drop, mlp_ratio=mlp_ratio, act_layer=act_layer,
            Close_attn=Close_attn
        ))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            layers.append(PPAFormer(dim=self.inplanes, num_heads=num_heads,
                               qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                               drop_path=drop_path_data[i],
                               norm_layer=norm_layer,
                               kernel_size=3, dilation=dilation,
                               drop=drop, mlp_ratio=mlp_ratio, act_layer=act_layer,
                               Close_attn=Close_attn
                               ))

        return nn.Sequential(*layers)

    def forward(self, x):
        stageData = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        stageData.append(x)

        x = self.layer2(x)
        stageData.append(x)

        return stageData


class Stageblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = StdConv2d(in_channels, out_channels, 1, 1)
        self.bn = nn.GroupNorm(32, out_channels, eps=1e-6)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out


class DFFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvdilateAttn4 = MSFA(1024)
        self.ConvdilateAttn3 = MSFA(512)

        self.Upcon = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)
        )

        self.Con1x1 = conv1x1(512, 1024)
        self.SoftPool2d = SoftPool2d(kernel_size=3, stride=2, padding=1)

        self.stage3con1x1 = conv1x1(1024, 512)
        self.stage4con1x1 = conv1x1(2048, 1024)

    def forward(self, stagex3, stagex4):
        stagex3_R = self.ConvdilateAttn3(stagex3)
        stagex3_C = self.Upcon(stagex4)

        stagex3_RC = stagex3_R + stagex3_C

        stagex4_R = self.ConvdilateAttn4(stagex4)
        stagex4_C1 = self.SoftPool2d(stagex3)
        stagex4_C = self.Con1x1(stagex4_C1)

        stagex4_RC = stagex4_R + stagex4_C

        out = self.stage3con1x1(torch.cat([stagex3_RC, self.Upcon(stagex4_RC)], dim=1))

        return out


class MFWF(nn.Module):
    def __init__(self, a=0.1, b=0.2):
        super().__init__()

        self.Parm1 = a
        self.Parm2 = b

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        )

        self.conT1 = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)

    def forward(self, x1, x2, x3):
        Out_x1 = self.upsample(x1)
        Out_x1 = self.conT1(Out_x1)

        Out_x2 = self.upsample(x2)
        Out_x2 = self.conT1(Out_x2)

        Out_x3 = self.upsample(x3)
        Out_x3 = self.conT1(Out_x3)

        out = self.Parm1 * Out_x1 + self.Parm2 * Out_x2 + (1 - self.Parm1 - self.Parm2) * Out_x3

        return out

class Decoder_block(nn.Module):
    def __init__(self):
        super().__init__()

        self.DFFM = DFFM()

        self.Conv1_1 = Stageblock(256 * 2, 256)
        self.Conv1_2 = Stageblock(256 * 3, 256)

        self.Conv0_1 = Stageblock(128 * 2, 128)
        self.Conv0_2 = Stageblock(128 * 3, 128)
        self.Conv0_3 = Stageblock(128 * 4, 128)

        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)
        )

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        )

        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        )

        self.MFWF = MFWF()

    def forward(self, x):
        x0_0 = x[0]
        x1_0 = x[1]
        x2_0 = x[2]
        x3_0 = x[3]

        x2_1 = self.DFFM(x2_0, x3_0)

        x1_1 = self.Conv1_1(torch.cat([x1_0, self.upsample2(x2_0)], dim=1))
        x1_2 = self.Conv1_2(torch.cat([x1_0, x1_1, self.upsample2(x2_1)], dim=1))

        x0_1 = self.Conv0_1(torch.cat([x0_0, self.upsample1(x1_0)], dim=1))
        x0_2 = self.Conv0_2(torch.cat([x0_0, x0_1, self.upsample1(x1_1)], dim=1))
        x0_3 = self.Conv0_3(torch.cat([x0_0, x0_1, x0_2, self.upsample1(x1_2)], dim=1))

        out = self.MFWF(x0_1, x0_2, x0_3)

        return out


class Decoder_block2(nn.Module):
    def __init__(self):
        super().__init__()

        self.DFFM = DFFM()

        self.Conv1_1 = Stageblock(256 * 2, 256)
        self.Conv1_2 = Stageblock(256 * 4, 256)

        self.Conv0_1 = Stageblock(128 * 2, 128)
        self.Conv0_2 = Stageblock(128 * 3, 128)
        self.Conv0_3 = Stageblock(128 * 5, 128)

        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)
        )

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        )

        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        )

        self.MFWF = MFWF()

    def forward(self, x, x2):
        x0_0 = x[0]
        x1_0 = x[1]
        x2_0 = x[2]
        x3_0 = x[3]

        out1 = x2[0]
        out2 = x2[1]

        x2_1 = self.DFFM(x2_0, x3_0)

        x1_1 = self.Conv1_1(torch.cat([x1_0, self.upsample2(x2_0)], dim=1))
        x1_2 = self.Conv1_2(torch.cat([out2, x1_0, x1_1, self.upsample2(x2_1)], dim=1))

        x0_1 = self.Conv0_1(torch.cat([x0_0, self.upsample1(x1_0)], dim=1))
        x0_2 = self.Conv0_2(torch.cat([x0_0, x0_1, self.upsample1(x1_1)], dim=1))
        x0_3 = self.Conv0_3(torch.cat([out1, x0_0, x0_1, x0_2, self.upsample1(x1_2)], dim=1))

        out = self.MFWF(x0_1, x0_2, x0_3)

        return out


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class ESMS_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_features = ResNet()
        self.Decoder_block = Decoder_block()
        self.Decoder_block2 = Decoder_block2()
        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=6,
            kernel_size=3,
        )

        self.Auxiliary_Enhanced_Encoder = Auxiliary_Enhanced_Encoder()

    def forward(self, x):
        out1 = self.forward_features(x)
        logics = self.Decoder_block(out1)
        preds = self.segmentation_head(logics)

        image2 = image_makeing(x, preds)

        out2 = self.Auxiliary_Enhanced_Encoder(image2)
        out2 = self.Decoder_block2(out1, out2)
        out = self.segmentation_head(out2)
        return out


if __name__ == '__main__':
    model = ESMS_Net()
    model.eval()
    print(model)
    image = torch.randn(1, 3, 256, 256)

    output = model(image)
    print("input:", image.shape)
    print("output:", output.shape)
