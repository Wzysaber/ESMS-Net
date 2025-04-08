import torch
import torch.nn as nn
import numpy as np
import time
import os
import logging

from configs.configs import parse_args

from model.CMT.CMT import CMT
from Network.ESMS_Net import ESMS_Net

from tool.train import close_optimizer
from tool.train import data_set
from tool.train import training
from tool.val import validating

from util.Loss import DiceLoss
from util.Data_process import Print_data
from util.Data_process import Creat_LineGraph

# 忽略相应的警告
import warnings

warnings.filterwarnings("ignore")

# 清除pytorch无用缓存
import gc


# # # 设置GPU的序列号
# import os
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "8"  # 设置采用的GPU序号


def main():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.set_device(0)

    # # 所以，当这个参数设置为True时，启动算法的前期会比较慢，但算法跑起来以后会非常快
    torch.backends.cudnn.benchmark = True

    # 导入配置
    args = parse_args()
    print(args.Information)
    # 加载训练和验证数据集
    train_loader = data_set(args)[0]
    train_dataset = data_set(args)[1]

    val_loader = data_set(args)[2]

    # 训练的相关配置
    device = torch.device(args.cuda)

    # 选择加载模型
    if args.model == "ESMS_Net":
        model = ESMS_Net()
    elif args.model == "CMT":
        model = CMT()


    if args.num_GPUs > 1:
        # 采用所有卡GPU服务器,
        # model = torch.nn.DataParallel(model).to(device)
        # 使用指定卡GPU
        print('num_GPUs:', args.num_GPUs)
        model = model.to(device)
        device_ids = [0, 1, 2, 7, 8]  # id显卡
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        print('num_GPUs:', args.num_GPUs)
        model = model.to(device)

    # 加载预训练模型
    if args.pretrained:
        print("loading pretrained model")
        old_dict = torch.load(args.pretrained_weights_path, map_location=device)['state_dict']
        model_dict = model.state_dict()

        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}

        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    # 加载中断模型
    if args.model_resume:
        print("Beginning resume model")
        loaded_state_dict = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(loaded_state_dict)

    # 损失函数
    criterion1 = nn.CrossEntropyLoss().to(device)
    criterion2 = DiceLoss(6).to(device)

    # 优化器选择
    optimizer = close_optimizer(args, model)

    # 将相应的参数进行打印
    Print_data(args.dataset_name, device, train_dataset.class_names,
               train_dataset, args.optimizer_name, args.model, args.total_epochs, args.sync_transform)

    # 训练及验证
    traincd_Data = []
    val_Data = []
    total_time = 0

    for epoch in range(args.start_epoch, args.total_epochs):
        since = time.time()
        loss_data = training(args, 6, model, optimizer, train_dataset, train_loader, criterion1, criterion2, device,
                             epoch)  # 对模型进行训练zzzz
        acc = validating(args, 6, model, optimizer, train_dataset, val_loader, device, epoch)  # 对模型进行验证

        # 记录时间
        time_elapsed = time.time() - since
        print('Time in  {:.0f}min:{:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        # 保存数据
        total_time += time_elapsed
        traincd_Data.append(loss_data)
        val_Data.append(acc)
        print(" ")

        if epoch in [50, 100, 150, 200, 250, 300]:
            torch.save(model.state_dict(),
                       os.path.join(args.directory, '{}epoch: {}.pth'.format(args.total_epochs, epoch)))

    # 记录总共时间
    print('Time in  {:.0f}h:{:.0f}min:{:.0f}s'.format(
        total_time // 3600, (total_time // 60) % 60, total_time % 60))
    Creat_LineGraph(val_Data, traincd_Data)  # 绘制相应曲线图

    # 保存最后一个epoch模型
    torch.save(model.state_dict(), os.path.join(args.directory, str(args.total_epochs) + ':last_model' + '.pth'))


if __name__ == "__main__":
    main()
