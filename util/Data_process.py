import os
import shutil
import torch
import matplotlib.pyplot as plt
from fvcore.nn import FlopCountAnalysis
from model.model_test.Ablation.Ablation1 import Ablation1
from model.model_test.Ablation.Ablation2 import Ablation2
from model.model_test.Ablation.Ablation3 import Ablation3
from model.model_test.Ablation.Ablation4 import Ablation4
from model.model_test.Ablation_Soft.Ablation2_Soft import Ablation2_Soft

from model.MARes_Unet.MaRes_Unet import MAResUNet
from model.DC_Swin.DC_Swin import dcswin_base
from model.CMT.CMT import cmt_b
from model.A2FCN.A2FCN import A2FPN
from model.model_test.DataChange_SMSNet.SMSNet_ting import SMSNet_ting
# 自定义类别
def fifteen_classes():
    return [
            'industrial land(IDL)',
            'urban residential(UR)',
            'rural residential(RR)',
            'traffic land(TL)',
            'paddy field(PF)',
            'irrigated land(IL)',
            'dry cropland(DC)',
            'garden plot(GP)',
            'arbor woodland(AW)',
            'shrub land(SL)',
            'natural grassland(NG)',
            'artificial grassland(AG)',
            'river(RV)',
            'lake(LK)',
            'pound(PN)',
            ]


def five_classes():
    return [
        '不透明表面',
        '建筑',
        '灌木',
        '树',
        '车',
    ]


def Print_data(dataset_name, device, class_name, train_dataset_len, optimizer_name, model, total_epochs,
               sync_transform):
    print('\ndataset:', dataset_name)
    print(device)
    print('classification:', class_name)
    print('Number samples {}.'.format(len(train_dataset_len)))  # 将模型的种类数和名称进行打印
    print('\noptimizer:', optimizer_name)
    print('model:', model)
    print('epoch:', total_epochs)

    if sync_transform:
        print("Have data Augmentation")
    else:
        print("NO data Augmentation")
    print("\nOK!,everything is fine,let's start training!\n")


def Creat_LineGraph(traincd_line, loss):
    x = range(len(traincd_line))
    y = traincd_line

    x_loss = range(len(loss))
    y_loss = loss

    plt.plot(x, y, color="g", label="ACC ", linewidth=0.7, marker=',')
    plt.plot(x_loss, y_loss, color="r", label=" loss", linewidth=0.7, marker=',')
    plt.xlabel('Epoch')
    plt.ylabel('data Value')
    plt.show()


def my_summary(test_model, H=256, W=256, C=3, N=2):
    model = test_model.cuda(6)
    print(model)
    inputs = torch.randn((N, C, H, W)).cuda(6)
    flops = FlopCountAnalysis(model, inputs)
    n_param = sum([p.nelement() for p in model.parameters()])
    print(f'GMac:{flops.total() / (1024 * 1024 * 1024)}')
    print(f'Params:{n_param}')


if __name__ == '__main__':
    model = Ablation1()
    resume_path = "/home0/students/master/2022/wangzy/Pycharm-Remote(161)/DT_model/weight/Vaihingen/Ablation1/11-24-10:50:23/219:  OA=0.8572 miou=0.7136 F1=0.8290.pth"
    loaded_state_dict = torch.load(resume_path)
    model.load_state_dict(loaded_state_dict)
    my_summary(model, 256, 256, 3, 2)
