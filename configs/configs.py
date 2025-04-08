import argparse
import time
import os
import json


# Potsdam_Img
# Vaihingen_Img
# 函数参数定义
def parse_args():
    parser = argparse.ArgumentParser(description="RemoteSensingSegmentation by PyTorch")
    # 'Baseline+PPAFormer+Auxiliary(01)+DFFM+(a：b: 1-a-b)解码阶段在2个阶段(0.1 0.2)(只含辅助解码)
    parser.add_argument('--Information', type=str,
                        default='Baseline+PPAFormer+Auxiliary+DFFM+MFWF')
    parser.add_argument('--dataset-name', type=str, default='Vaihingen')
    parser.add_argument('--model', type=str, default='ESMS_Net', help='model name')
    parser.add_argument('--cuda', type=str, default="cuda:0")
    parser.add_argument('--sync_transform', default=True)

    parser.add_argument('--total-epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 120)')

    parser.add_argument('--save-file', default=True)

    parser.add_argument('--optimizer-name', type=str, default='SGD')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default:0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M', help='weight-decay (default:1e-4)')
    parser.add_argument('--base-lr', type=float, default=0.01, metavar='M', help='')

    parser.add_argument('--best-miou', type=float, default=0)
    parser.add_argument('--best-OA', type=float, default=0)
    parser.add_argument('--best-F1', type=float, default=0)

    # dataset
    parser.add_argument('--train-data-root', type=str,
                        default="/home0/students/master/2022/wangzy/Pycharm-Remote(161)/DT_model/Vaihingen_Img/Train/")
    parser.add_argument('--val-data-root', type=str,
                        default="/home0/students/master/2022/wangzy/Pycharm-Remote(161)/DT_model/Vaihingen_Img/Test/")
    parser.add_argument('--train-batch-size', type=int, default=4, metavar='N',
                        help='batch size for training (default:16)')
    parser.add_argument('--val-batch-size', type=int, default=4, metavar='N',
                        help='batch size for testing (default:16)')

    # output_save_path
    # strftime格式化时间，显示当前的时间
    parser.add_argument('--experiment-start-time', type=str,
                        default=time.strftime('%m-%d-%H:%M:%S', time.localtime(time.time())))
    parser.add_argument('--save-pseudo-data-path', type=str,
                        default='/home/students/master/2022/wangzy/PyCharm-Remote/ST_Unet_test/pseudo_data')
    parser.add_argument('--Delmodel-name', type=str, default='wzy')

    # augmentation
    parser.add_argument('--base-size', type=int, default=256, help='base image size')
    parser.add_argument('--crop-size', type=int, default=256, help='crop image size')
    parser.add_argument('--flip-ratio', type=float, default=0.5)
    parser.add_argument('--resize-scale-range', type=str, default='0.5, 2.0')

    # model
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--pretrained_weights_path', type=str,
                        default="/home0/students/master/2022/wangzy/Pycharm-Remote(161)/DT_model/pretrained/stseg_tiny.pth")
    parser.add_argument('--resume-path', type=str,
                        default="/home0/students/master/2022/wangzy/Pycharm-Remote(161)/DT_model/weight/Potsdam/Ablation1/07-31-11:00:59/95:  OA=0.8775 miou=0.7658 F1=0.8657.pth")
    parser.add_argument('--model-resume', type=str, default=False)

    # criterion
    # 损失的权重值
    parser.add_argument('--class-loss-weight', type=list, default=
    [0.008728536232175135, 0.05870821984204281, 0.030766985878693004, 0.03295408432939304, 0.2399409412190348,
     0.20305583055639448, 0.6344888568739531, 0.16440413437125656, 0.5372260524694122, 0.22310945250778813,
     0.04659596810284655, 0.19246378709444723, 0.6087430986295436, 0.34431415558778183, 0.4718853977371564, 1.0])

    # loss
    parser.add_argument('--loss-names', type=str, default='cross_entropy')
    parser.add_argument('--classes-weight', type=str, default=None)

    # environment
    parser.add_argument('--use-cuda', action='store_true', default=True, help='using CUDA training')
    parser.add_argument('--num-GPUs', type=int, default=1, help='numbers of GPUs')
    parser.add_argument('--num_workers', type=int, default=32)

    # validation
    parser.add_argument('--eval', action='store_true', default=False, help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default=False)

    parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='start epoch (default:0)')

    args = parser.parse_args()

    directory = "weight/%s/%s/%s/" % (args.dataset_name, args.model, args.experiment_start_time)
    args.directory = directory

    if args.save_file:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("Creat and Save model.pth!")

            config_file = os.path.join(directory, 'config.json')
            with open(config_file, 'w') as file:
                json.dump(vars(args), file, indent=4)

    return args
