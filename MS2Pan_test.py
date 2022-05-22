import torch.nn.modules as nn
import torch
import numpy as np
from PhysicPanModels import MS2PAN
import h5py
import scipy.io as sio
import os
import statistics
from PIL import Image
from util import *

sensor = 'WV3'
ms_channels = 8 if sensor == "WV3" else 4
pix_max_val = 1023. if sensor == "GF2" else 2047.

def load_set(file_path):
    ## ===== case1: NxCxHxW
    data = h5py.File(file_path)
    dataset_names = list(data.keys())

    # tensor type:
    hms1 = data['HRMSS'][...]  # NxCxHxW
    hms1 = np.array(hms1, dtype=np.float32) / pix_max_val
    hms1 = hms1.transpose(3, 2, 0, 1)
    hms = torch.from_numpy(hms1)

    pan1 = data['PANS'][...]  # NxCxHxW
    pan1 = np.array(pan1, dtype=np.float32) / pix_max_val
    pan1 = pan1.transpose(3, 2, 0, 1)
    pan = torch.from_numpy(pan1)

    return hms, pan

# ==============  Main test  ================== #
ckpt = "pretrained/ms2pan%s.pth" % (sensor)

def test(file_path):
    hms, pan = load_set(file_path)
    model = MS2PAN(ms_channels).cuda().eval()
    weight = torch.load(ckpt)
    model.load_state_dict(weight)

    with torch.no_grad():

        x1, x2 = hms, pan   # read data: CxHxW (numpy type)
        print(x1.shape)
        x1 = x1.cuda().float()  # convert to tensor type:
        x2 = x2.cuda().float()  # convert to tensor type:

        sr = model(x1)  # tensor type: sr = NxCxHxW

        print(sr.shape)

        pan = pan.type(torch.FloatTensor)
        num_exm = sr.shape[0]

        psnrs = []
        ssims = []
        for index in range(num_exm):  # save the DL results to the 03-Comparisons(Matlab)
            ## show images
            nparr1 = pan[index, 0, :, :]
            img10 = tensor2img(nparr1)

            nparr2 = sr[index, 0, :, :]
            img20 = tensor2img(nparr2)

            psnr = calculate_psnr(img10, img20)
            ssim = calculate_ssim(img10, img20)
            psnrs.append(psnr)
            ssims.append(ssim)
            print("PSNR:{:.4f}, SSIM:{:.4f}".format(psnr, ssim))

        print("psnr: %.2f%s%.2f, ssim: %.2f%s%.2f" % (statistics.mean(psnrs), "\u00B1", statistics.stdev(psnrs), statistics.mean(ssims), "\u00B1", statistics.stdev(ssims)))


###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == '__main__':

    file_path = "../data/tst(%s)_0064-0064.h5" % (sensor)

    test(file_path)
