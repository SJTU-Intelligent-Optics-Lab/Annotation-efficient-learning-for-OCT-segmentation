import argparse
import os
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
import networks.unetr
from utils.trainer import run_training, val_epoch
from utils.data_utils import get_loader
from networks.unet import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction
import random

def main(train,val):
    parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
    parser.add_argument("--test_mode", default=False, type=bool, help="test mode or not")
    parser.add_argument("--label_index",default=[255],type=list, help="index of label for foreground segmentation")
    parser.add_argument("--phase2_weight", default="./runs/phase2 model/phase2 model.pth", type=str,help="load phase2 model weight")
    parser.add_argument("--phase1_weight", default="./runs/phase1 model/phase1 model.pth", type=str, help="load phase1 model weight")
    parser.add_argument("--logdir", default="./runs/log", type=str, help="directory to save the tensorboard logs")
    parser.add_argument('--list_dir', type=str, default='./dataset', help='list dir')
    parser.add_argument("--data_dir", default="./dataset", type=str, help="dataset directory")
    parser.add_argument("--train_text", default=train, type=str, help="training image list text name")
    parser.add_argument("--val_text", default=val, type=str, help="validating image list text name")
    parser.add_argument('--model_name', default='vit_large_patch16', type=str, metavar='MODEL',help='Name of model to train')
    parser.add_argument("--save_checkpoint", default=True,type=bool, help="save checkpoint during training")
    parser.add_argument("--max_epochs", default=100, type=int, help="max number of training epochs")
    parser.add_argument("--batch_size", default=4, type=int, help="number of batch size")
    parser.add_argument("--optim_lr", default=0.5*1e-4, type=float, help="optimization learning rate")
    parser.add_argument("--optim_name", default="adam", type=str, help="optimization algorithm")#adamw
    parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
    parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
    parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
    parser.add_argument("--val_every", default=1, type=int, help="validation frequency")
    parser.add_argument("--distributed", action="store_true", help="start distributed training")
    parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--num_workers", default=2, type=int, help="number of workers")
    parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
    parser.add_argument("--feature_size", default=64, type=int, help="feature size dimention")#
    parser.add_argument("--in_channels", default=3, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=1, type=int, help="number of output channels")
    parser.add_argument("--res_block", default=True,type=bool, help="use residual blocks")
    parser.add_argument("--conv_block", default=True,type=bool, help="use conv blocks")
    parser.add_argument('--input_size', default=224, type=int,help='images input size')#
    parser.add_argument("--dropout_rate", default=0.2, type=float, help="dropout rate")
    parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
    parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
    parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")

    args = parser.parse_args()
    args.amp = not args.noamp

    main_worker(gpu=0, args=args)
    seed = 1244
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main_worker(gpu, args):
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True

    loader = get_loader(args)
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)


    if (args.model_name is None) or args.model_name in ['vit_base_patch16','vit_large_patch16','vit_huge_patch14']:
        model = networks.unetr.__dict__[args.model_name](
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=args.input_size,
            feature_size=args.feature_size,
            norm_name=args.norm_name,
            conv_block=True,
            res_block=True,
            dropout_rate=args.dropout_rate,
            num_classes=1000)

        # loading weights
        if not args.test_mode:
            if args.phase1_weight is not None and args.phase2_weight is None:# training phase 2
                checkpoint = torch.load(args.phase1_weight,map_location='cpu')
                checkpoint_model = checkpoint['model']
                print(checkpoint['model'])
                model.load_state_dict(checkpoint_model, strict=False)
                print("Use pretrained weights")
                #print(model.state_dict())
            elif args.phase1_weight is not None and args.phase2_weight is not None: # training phase 4
                #load weight of phase2 model first (for decoder)
                assert args.phase2_weight != None, 'No segmentation model weights loaded'
                model_dict = torch.load(args.phase2_weight, map_location="cpu")
                print(model_dict['state_dict'])
                model.load_state_dict(model_dict['state_dict'], strict=True)
                # then load weight of phase1 model (for encoder)
                checkpoint = torch.load(args.phase1_weight,map_location='cpu')
                checkpoint_model = checkpoint['model']
                print(checkpoint_model)
                model.load_state_dict(checkpoint_model, strict=False)


        else:# test mode, load best model
                assert args.phase2_weight != None, 'No segmentation model weights loaded'
                model_dict = torch.load(args.phase2_weight, map_location="cpu")
                model.load_state_dict(model_dict['state_dict'])

    elif args.model_name =='unet':
        model = UNet(n_channels = args.in_channels, n_classes = args.out_channels, bilinear=True)
    else:
        raise ValueError("Unsupported model " + str(args.model_name))

    #setup loss
    dice_loss = DiceCELoss(
        to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr)

    post_label = AsDiscrete(to_onehot=True, n_classes=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=args.out_channels)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    best_acc = 0
    start_epoch = 0

    model.cuda(args.gpu)

    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1,
                                                           patience=5)  # goal: maximize Dice score

    if args.model_name =='unet':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.optim_lr, weight_decay=1e-8, momentum=0.99)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)  # goal: maximize Dice score

    if not args.test_mode:
        accuracy = run_training(
            model=model,
            train_loader=loader[0],
            val_loader=loader[1],
            optimizer=optimizer,
            loss_func=dice_loss,
            acc_func=dice_acc,
            args=args,
            model_inferer=None,
            scheduler=scheduler,
            start_epoch=start_epoch,
            post_label=post_label,
            post_pred=post_pred,
        )
    else:
        accuracy, run_loss = val_epoch(
            model=model,
            loader=loader,
            args=args
        )
        print("final acc:", accuracy, 'loss:', run_loss)
    return accuracy


if __name__ == "__main__":

    train = 'train_fewshot_data'
    val = 'val_fewshot_data'
    main(train,val)
