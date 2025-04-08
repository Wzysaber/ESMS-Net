import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
# from model.Unet.Unet import Unet
from model.model_test.model_test20.model_test20 import model_test20
from model.model_test.Ablation.Ablation1 import Ablation1
from model.model_test.Ablation.Ablation2 import Ablation2
from model.model_test.Ablation.Ablation3 import Ablation3
from model.model_test.Ablation.Ablation4 import Ablation4

from model.model_test.Ablation_Soft.Ablation1_Soft import Ablation1_Soft
from model.model_test.Ablation_Soft.Ablation2_Soft import Ablation2_Soft
from model.model_test.Ablation_Soft.Ablation3_Soft import Ablation3_Soft

from model.A2FCN.A2FCN import A2FPN
from model.CMT.CMT import cmt_b
from model.Unet.Unet import Unet

import os

from model.UNetFormer.UNetFormer import UNetFormer

from util.palette import colorize_mask
from util.palette import colorize_mask2
from Parameter import metric
from prettytable import PrettyTable
from collections import OrderedDict

import matplotlib.pyplot as plt


# 定义预测函数
def predict(model, image_path):
    # 加载图像并做相应预处理
    img = Image.open(image_path).convert('RGB')

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img).to(device)
    img = img.unsqueeze(0)  # 增加batch维
    img = torch.cat([img, img], dim=0)

    # 对输入图像进行预测
    output = model(img)
    pred = output.argmax(dim=1)  # 取最大值的索引
    # _, pred = torch.max(output, 1)  # 加_,则返回一行中最大数的位置。
    pred = pred[1]
    # 转为numpy数组并去掉batch维
    pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)  # 将数据提取出来

    return pred


if __name__ == '__main__':
    # 加载相应参数
    device = torch.device("cuda:0")
    choose = 2

    if choose == 1:
        path = "/home0/students/master/2022/wangzy/predict_Vimage/Ablation_pic/Mask_image/1.jpg"
    else:
        path = "/home0/students/master/2022/wangzy/predict_Vimage/Ablation_pic/image/2.jpg"

    image_path = path
    model_path = ""

    # 加载原始标签
    image = cv2.imread(image_path)

    # 加载模型
    model = Unet(num_classes=6)
    model.eval()
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)

    # 预测图像
    pred = predict(model, image_path)
    overlap = colorize_mask(pred)
    overlap2 = colorize_mask2(pred)

    # 可视化预测结果

    plt.imshow(overlap)
    plt.axis('off')  # 去除坐标轴
    # plt.savefig("/home0/students/master/2022/wangzy/OR_image/TGRS/SMS-Net/1.png", bbox_inches='tight')
    plt.show()

    plt.imshow(overlap2)
    plt.axis('off')  # 去除坐标轴
    # plt.savefig("/home0/students/master/2022/wangzy/OR_image/TGRS/SMS-Net/1.png", bbox_inches='tight')
    plt.show()

    plt.title("label")
    plt.imshow(image)
    plt.show()
