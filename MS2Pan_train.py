import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from DatasetProc import *
from torch.utils.data import DataLoader
from PhysicPanModels import MS2PAN_Simpler
import copy
from torch.utils.tensorboard import SummaryWriter
from util import *

# ============= 2) HYPER PARAMS(Pre-Defined) ==========#
lr = 0.0003
epochs = 2500 #450
ckpt = 50
batch_size = 32
set_random_seed(126)
sensor = 'GF2'
ms_channels = 8 if sensor == "WV3" else 4
pix_max_val = 1023. if sensor == "GF2" else 2047.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# device = torch.device("cuda:1")
model = MS2PAN_Simpler(ms_channels).to(device)

loss_func = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
best_weights = copy.deepcopy(model.state_dict())

writer = SummaryWriter('./train_logs')
model_path = 'Weights({})_MS2Pan'.format(sensor)
mkdir(model_path)
def save_checkpoint(model, epoch):  # save model function
    model_out_path = model_path + '/' + "ms2pan{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
def load_checkpoint(model, epoch):  # save model function
    model_out_path = model_path + '/' + "ms2pan{}.pth".format(epoch)
    if os.path.exists(model_out_path):
        checkpoint = torch.load(model_out_path)
        model.load_state_dict(checkpoint)
        return True
    return False
###################################################################
# ------------------- Main Train (Run second)----------------------
###################################################################
def train(training_data_loader, validate_data_loader,start_epoch=0):
    print('Start training...')
    best_epoch = 0
    best_psnr = 0.0
    if start_epoch > 0 and load_checkpoint(model, start_epoch) == False:
        start_epoch = 0
    for epoch in range(start_epoch, epochs, 1):
        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []

        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            pan, hms = batch[0].to(device), batch[2].to(device)

            optimizer.zero_grad()  # fixed

            res = model(hms)  # call model

            loss = loss_func(res, pan)  # compute loss
            epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch

            loss.backward()  # fixed
            optimizer.step()  # fixed


        t_loss_mean = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        writer.add_scalar('mse_loss/t_loss_mean', t_loss_mean, epoch)  # write to tensorboard to check
        print('Epoch: {}/{} training loss: {:.7f}'.format(epoch, epochs, t_loss_mean))  # print loss for each epoch

        if epoch % ckpt == 0:  # if each ckpt epochs, then start to save model
            save_checkpoint(model, epoch)

        # ============Epoch Validate=============== #
        model.eval()
        epoch_psnr = AverageMeter()

        with torch.no_grad():  # fixed
            for iteration, batch in enumerate(validate_data_loader, 1):
                pan, hms = batch[0].to(device), batch[2].to(device)

                fake_pan = model(hms)  # call model

                loss = loss_func(fake_pan, pan)
                epoch_val_loss.append(loss.item())

                epoch_psnr.update(calc_psnr(pan, fake_pan), len(pan))
            v_loss_mean = np.nanmean(np.array(epoch_val_loss))
            writer.add_scalar('val/v_loss_mean', v_loss_mean, epoch)
            print('Epoch: {}/{} validate loss: {:.7f}'.format(epoch, epochs, v_loss_mean))  # print loss for each epoch
            print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
        if epoch_psnr.avg > best_psnr:
            best_psnr = epoch_psnr.avg
            print('best_epoch: {:d}'.format(epoch))
            save_checkpoint(model, 'Best')

    writer.close()  # close tensorboard

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
if __name__ == "__main__":

    train_set = DatasetProc4Physic('../data/new_tra(GF2)_0032-0032.h5', pix_max_val)  # creat data for training

    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True, worker_init_fn=seed_worker)

    validate_set = DatasetProc4Physic('../data/new_val(GF2)_0032-0032.h5', pix_max_val)  # creat data for validation
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=False,
                                      pin_memory=True, drop_last=True, worker_init_fn=seed_worker)

    train(training_data_loader, validate_data_loader, 0)