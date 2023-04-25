import os
import shutil
import time
import cv2
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.modules.loss import BCELoss

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class DiceLoss(torch.nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        else:
            inputs = torch.sigmoid(inputs)
        #target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    smooth = 1e-5
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return (2 * intersect +smooth) / (x_sum + y_sum + smooth)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()

    ce_loss = torch.nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss(1)
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):

        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]

        if args.in_channels == 1:
            data = torch.unsqueeze(data[:, 0, :, :], dim=1)

        data, target = data.cuda(args.rank), target.cuda(args.rank)
        for param in model.parameters():
            param.grad = None

        with autocast(enabled=args.amp):
            logits = model(data)

            loss_ce = ce_loss(logits, target)
            loss_dice = dice_loss(logits, target, softmax=False)
            loss = 1.0 * loss_ce + 1.0 * loss_dice
            print('ce_loss: {:.3f} dice_loss: {:.3f}'.format(loss_ce, loss_dice))

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(loss.item()),
                "lr: {:.8f}".format(optimizer.param_groups[0]['lr']),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None

    return run_loss.avg


def val_epoch(model, loader, epoch=None, acc_func=None, args=None, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()
    ce_loss = torch.nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss(1)
    run_loss = AverageMeter()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target, image_path, h, w = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]

            if args.in_channels == 1:
                data = torch.unsqueeze(data[:, 0, :, :], dim=1)  # 三通道改单通道

            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
                    #loss
                    loss_ce = ce_loss(logits, target)
                    loss_dice = dice_loss(logits, target, softmax=False)
                    loss = 1.0 * loss_ce + 1.0 * loss_dice
                    #print('ce_loss: {:.3f} dice_loss: {:.3f}'.format(loss_ce, loss_dice))
                    #sigmoid for dice
                    out = torch.sigmoid(logits)
            acc_list = []
            if out.is_cuda:
                target = target.cpu().numpy()
                out = out.cpu().detach().numpy()
                h = h.cpu().numpy()
                w = w.cpu().numpy()
                assert out.shape == target.shape, 'predict {} & target {} shape do not match'.format(out.shape, target.shape)
                for i in range(out.shape[0]):
                    out[i] = np.where(out[i]>0.6, 1, 0)

                    acc_list.append(dice(out[i],target[i]))
                    pre = np.array(out[i],dtype=np.uint8)
                    pre = np.squeeze(pre,axis=0)
                    pre = cv2.resize(pre,(w[i], h[i]),interpolation=cv2.INTER_NEAREST)*255
                    #postprocess
                    # pre[0:1,:]=0
                    # pre[:, 0:1] = 0
                    # pre = fillHole(pre)
                    # pre = save_max_objects(pre)


                    pre_path = image_path[i].replace('val_fewshot_data', 'val_fewshot_results')
                    if not os.path.isdir(os.path.split(pre_path)[0]):
                        os.makedirs(os.path.split(pre_path)[0])

                    path=pre_path.replace('.png', '_auto.png')
                    cv2.imwrite(path, pre)


            avg_acc = np.mean(np.array(acc_list))
            run_acc.update(avg_acc, n=args.batch_size)
            run_loss.update(loss.item(), n=args.batch_size)
            if args.rank == 0:
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "dice:", avg_acc,
                    'loss:',loss.cpu().numpy(),
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
    return run_acc.avg, run_loss.avg, os.path.split(pre_path)[0]


def save_checkpoint(model, epoch, args, filename="model.pth", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    early_stopping = EarlyStopping(patience=10, verbose=True)
    spend_time = 0
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()#using float16 to reduce memory
    val_acc_max = 0.0
    for epoch in range(start_epoch, args.max_epochs):

        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )#for training one epoch

        # if scheduler is not None:
        #     scheduler.step()

        spend_time += time.time() - epoch_time
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
            with open(os.path.join(args.logdir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write("Final training:{}/{},".format(epoch, args.max_epochs - 1) + "loss:{}".format(train_loss) + "\n")
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc, run_loss, file_path = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )
            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc:", val_avg_acc,
                    'std:',
                    'loss:',run_loss,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                with open(os.path.join(args.logdir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write("Final validation:{}/{},".format(epoch, args.max_epochs - 1) + "dice:{},".format(val_avg_acc)
                            + "loss:{},".format(run_loss)+ "\n")
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    #fewshot save bset results
                    if os.path.exists(file_path.replace('val_fewshot_results','val_fewshot_results_best')):
                        shutil.rmtree(file_path.replace('val_fewshot_results','val_fewshot_results_best'))
                    shutil.copytree(file_path,file_path.replace('val_fewshot_results','val_fewshot_results_best'))

                    #unet save best results
                    if os.path.exists('./dataset/val_fewshot_results_best'):
                        shutil.rmtree('./dataset/val_fewshot_results_best')
                    shutil.copytree('./dataset/val_fewshot_results',
                                    './dataset/val_fewshot_results_best')

                    #save weights
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pth")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pth"), os.path.join(args.logdir, "model.pth"))

            early_stopping(val_avg_acc)
            if early_stopping.early_stop:
                print("Early stop！")
                break

            if scheduler is not None:
                scheduler.step(-val_avg_acc)

    print("Training Finished !, Best Accuracy: ", val_acc_max, "Total time: {} s.".format(round(spend_time)))

    return val_acc_max

from skimage import measure
def save_max_objects(img):
    labels = measure.label(img, connectivity=1)
    jj = measure.regionprops(labels)
    # is_del = False
    if len(jj) == 0:
        out = img
        return out
    elif len(jj) == 1:
        out = img
        return out
        # is_del = False
    else:
        num = labels.max()
        del_array = np.array([0] * (num + 1))
        for k in range(num):
            if k == 0:
                initial_area = jj[0].area
                save_index = 1
            else:
                k_area = jj[k].area

                if initial_area < k_area:
                    initial_area = k_area
                    save_index = k + 1

        del_array[save_index] = 1
        del_mask = del_array[labels]
        out = img * del_mask
        return out


def fillHole(im_in):
    im_in = im_in.astype(np.uint8)
    # print np.unique(im_in)
    im_floodfill = im_in.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_in | im_floodfill_inv
    # print np.unique(im_out)
    return im_out
