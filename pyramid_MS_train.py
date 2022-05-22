import os
import time
import datetime
import torch
import torch.nn as nn
import sys

# this is a pointer to the module object instance itself.
this = sys.modules[__name__]
from DatasetProc import *
from torch.utils.data import DataLoader
from models import *

from torch.utils.tensorboard import SummaryWriter
from util import *
from PhysicPanModels import *

# ============= 2) HYPER PARAMS(Pre-Defined) ==========#
set_random_seed(911)
lr_start = 0.0003
epochs = 1000
ckpt = 50
batch_size = 32
down_scales = 3

this.weight_dir = None
sensor = 'WV3'
ms_channels = 8 if sensor == "WV3" else 4
pix_max_val = 1023. if sensor == "GF2" else 2047.

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print_gpu_status(device)

model = PyramidInjection(ms_channels, down_scales).to(device)
total_params2learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters:{}".format(total_params2learn))

model_ms2pan = MS2PAN(ms_channels).to(device)
weight_ms2pan = torch.load("pretrained/ms2pan"+sensor+".pth", map_location=device)
model_ms2pan.load_state_dict(weight_ms2pan)

model_hr2lr = HR2LRMS(ms_channels).to(device)
weight_hr2lr = torch.load("pretrained/hr2lr"+sensor+".pth", map_location=device)
model_hr2lr.load_state_dict(weight_hr2lr)

loss_L1 = nn.SmoothL1Loss(reduction='mean')
loss_L2 = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)

def save_checkpoint(model, epoch, optimer):  # save model function
    model_out_path = this.weight_dir + '/' + "{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
def load_checkpoint(model, epoch):  # save model function
    model_out_path = this.weight_dir + '/' + "{}.pth".format(epoch)
    if os.path.exists(model_out_path):
        checkpoint = torch.load(model_out_path)
        model.load_state_dict(checkpoint)
        return True
    return


def adjust_learning_rate(_optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_start * (0.9 ** (epoch // 300))
    for param_group in _optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(_optimizer):
    lr = []
    for param_group in _optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def adjust_coef(coefInit, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    curr_coef = coefInit * (2. * (epoch // 200))
    return curr_coef


def get_loss(loss_mode, res, gt, pan, lms):
    loss = loss_L2(res, gt)  # if lossMode == 0:
    loss_ms2pan = torch.tensor(0.)
    loss_ms2ls = torch.tensor(0.)
    if loss_mode == 1: # MS2PAN
        fake_pan = model_ms2pan(res)  # tensor type: res = NxCxHxW
        fake_pan = fake_pan.clip(0, 1)
        loss_ms2pan = loss_L1(fake_pan, pan)
    elif loss_mode == 2:
        fake_lms = model_hr2lr(res)  #  fake_hms = res
        loss_ms2ls = loss_L1(fake_lms, lms)
    elif loss_mode == 3:
        fake_pan = model_ms2pan(res)  # tensor type: res = NxCxHxW
        loss_ms2pan = loss_L1(fake_pan, pan)

        fake_lms = model_hr2lr(res)  # fake_hms = res
        loss_ms2ls = loss_L1(fake_lms, lms)
    
    return loss, loss_ms2pan, loss_ms2ls
###################################################################
# ------------------- Main Train (Run second)----------------------
###################################################################


def train(training_data_loader, validate_data_loader, loss_mode, coef_extra=0.02, alpha4hr2lr=0.8, start_epoch=0):

    print('Start training...')
    best_epoch = 0
    best_ergas = float("inf")

    this.weight_dir = '%s(%s)_Mode%d(sp%d)_lr%.4f_Epo%d_Bat%d_Loss%s_%s' % (
        sensor, type(model).__name__, loss_mode, down_scales, lr_start, epochs, batch_size,
        type(loss_func).__name__, optimizer.__class__.__name__)
    if loss_mode != 0:
        this.weight_dir = this.weight_dir + '_(%g,%g)' % (coef_extra, alpha4hr2lr)
    mkdir(this.weight_dir)
    print(this.weight_dir)

    writer = SummaryWriter('./train_logs', filename_suffix=this.weight_dir)

    prev_time = time.time()

    if start_epoch > 0 and load_checkpoint(model, start_epoch) is False:
        start_epoch = 0
    print('starting epoch:{}'.format(start_epoch))
    print('mode:{:d}, coef:{}, alpha:{}'.format(loss_mode, coef_extra, alpha4hr2lr))

    for epoch in range(start_epoch, epochs, 1):

        adjust_learning_rate(optimizer, epoch)
        current_lr = get_learning_rate(optimizer)[0]
        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []
        epoch_train_ms2pan_loss, epoch_train_hr2lr_loss = [], []
        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            pan, lms, goal, ums = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)

            optimizer.zero_grad()  # fixed

            res = model(ums, pan)  # call model

            # calculate loss
            loss, loss_ms2pan, loss_hr2lr = get_loss(loss_mode, res, goal, pan, lms)
            loss_extra = (1. - alpha4hr2lr) * loss_ms2pan + alpha4hr2lr * loss_hr2lr
            loss = loss + coef_extra * loss_extra
            
            epoch_train_ms2pan_loss.append(loss_ms2pan.item())
            epoch_train_hr2lr_loss.append(loss_hr2lr.item())
            epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch
    
            loss.backward()  # fixed
            optimizer.step()  # fixed


        t_loss_mean = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of entire losses, as one epoch loss
        t_ms2pan_loss_mean = np.nanmean(np.array(epoch_train_ms2pan_loss))  # compute the mean value of ms2pan losses, as one epoch loss
        t_hr2lr_loss_mean = np.nanmean(np.array(epoch_train_hr2lr_loss))  # compute the mean value of hr2lr losses, as one epoch loss
        writer.add_scalar('mse_loss/t_loss_mean', t_loss_mean, epoch)  # write to tensorboard to check

        batches_done = epoch - start_epoch
        batches_left = epochs - start_epoch - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        print('Epoch: {}/{}, lr:{}, coef:{:.3f}, training loss: {:.6f}/{:.6f}/{:.6f}, ETA: {}'.format(epoch, epochs, current_lr,
                                                                                      coef_extra, t_loss_mean,
                                                                                      t_ms2pan_loss_mean, t_hr2lr_loss_mean,
                                                                                      time_left))  # print loss for each epoch
        if epoch % ckpt == 0:  # if each ckpt epochs, then start to save model
            save_checkpoint(model, epoch, optimizer)

        # ============Epoch Validate=============== #
        model.eval()
        epoch_ergas = AverageMeter()

        with torch.no_grad():  # fixed
            for iteration, batch in enumerate(validate_data_loader, 1):
                pan, lms, goal, ums = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)

                res = model(ums, pan)  # call model
                loss, loss_ms2pan, loss_hr2lr = get_loss(loss_mode, res, goal, pan, lms)
            
                loss_extra = (1. - alpha4hr2lr) * loss_ms2pan + alpha4hr2lr * loss_hr2lr
                loss = loss + coef_extra * loss_extra
                epoch_val_loss.append(loss.item())
                for idx in range(batch_size):
                    epoch_ergas.update(calculate_ergas(res[idx, ].permute(1, 2, 0).cpu().numpy(),
                                                       goal[idx, ].permute(1, 2, 0).cpu().numpy()), 1)
            v_loss_mean = np.nanmean(np.array(epoch_val_loss))
            writer.add_scalar('val/v_loss_mean', v_loss_mean, epoch)

            v_ergas_mean = epoch_ergas.avg
            writer.add_scalar('val/v_ergas_mean', v_ergas_mean, epoch)

            print('Epoch: {}/{} validate loss: {:.7f}'.format(epoch, epochs, v_loss_mean))  # print loss for each epoch
            print('eval ergas: {:.3f}'.format(epoch_ergas.avg))
        if epoch_ergas.avg < best_ergas:
            best_ergas = epoch_ergas.avg
            print('best_epoch: {:d}'.format(epoch))
            save_checkpoint(model, 'best', optimizer)

    writer.close()  # close tensorboard


if __name__ == "__main__":
    train_set = DatasetProc4Physic('../data/tra(%s)_0032-0032.h5' % (sensor), pix_max_val)  # creat data for training
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                  pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    validate_set = DatasetProc4Physic('../data/val(%s)_0032-0032.h5' % (sensor), pix_max_val)  # creat data for validation
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=False,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    train(training_data_loader, validate_data_loader, loss_mode=3, coef_extra=0.03, alpha4hr2lr=0.4,
          start_epoch=0)

