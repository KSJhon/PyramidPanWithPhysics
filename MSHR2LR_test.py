import torch.nn.modules as nn
import torch
import numpy as np
from PhysicPanModels import HR2LRMS
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
    data = h5py.File(file_path)
    dataset_names = list(data.keys())

    # tensor type:
    lms1 = data['LRMSS'][...]  # NxCxHxW
    lms1 = np.array(lms1, dtype=np.float32) / pix_max_val
    lms1 = lms1.transpose(3, 2, 0, 1)
    lms = torch.from_numpy(lms1)

    hms1 = data['HRMSS'][...]  # NxCxHxW
    hms1 = np.array(hms1, dtype=np.float32) / pix_max_val
    hms1 = hms1.transpose(3, 2, 0, 1)
    hms = torch.from_numpy(hms1)

    return lms, hms


# ==============  Main test  ================== #

ckpt = "pretrained/hr2lrWV3.pth"
# sam: 0.93±0.49, ergas: 1.26±0.66
# psnr: 42.59±5.47, ssim: 0.99±0.01
# ckpt = "pretrained/hr2lrQB.pth"
# sam: 0.36±0.20, ergas: 0.36±0.23
# psnr: 54.43±6.89, ssim: 1.00±0.01
# ckpt = "pretrained/hr2lrWV2.pth"
# sam: 0.11±0.26, ergas: 0.13±0.14
# psnr: 62.78±5.24, ssim: 1.00±0.00
# ckpt = "pretrained/hr2lrGF2.pth"
# sam: 0.44±0.22, ergas: 0.56±0.30
# psnr: 46.19±4.98, ssim: 0.99±0.01


def test(file_path):
    lms, hms = load_set(file_path)

    model = HR2LRMS(ms_channels).cuda().eval()
    weight = torch.load(ckpt)
    model.load_state_dict(weight)

    with torch.no_grad():

        x1, x2 = hms, lms   # read data
        x1 = x1.cuda().float()  # convert to tensor type:
        x2 = x2.cuda().float()  # convert to tensor type:

        num_exm = x1.shape[0]

        ergas = []
        sam = []
        psnr = []
        ssim = []
        for index in range(num_exm):

            res = model(x1[index].unsqueeze(0))

            psnr.append(calculate_psnr(res.squeeze().permute(1, 2, 0).cpu().numpy(), x2[index, :, :, :].permute(1, 2, 0).cpu().numpy(), dynamic_range=1))
            ssim.append(calculate_ssim(res.squeeze().permute(1, 2, 0).cpu().numpy(), x2[index, :, :, :].permute(1, 2, 0).cpu().numpy(), dynamic_range=1))
            ergas.append(calculate_ergas(res.squeeze().permute(1, 2, 0).cpu().numpy(),
                                         x2[index, :, :, :].permute(1, 2, 0).cpu().numpy()))
            sam.append(calculate_sam(res.squeeze().permute(1, 2, 0).cpu().numpy(),
                                     x2[index, :, :, :].permute(1, 2, 0).cpu().numpy()) * 180 / math.pi)

        print("sam: %.2f%s%.2f, ergas: %.2f%s%.2f" % (
                statistics.mean(sam), "\u00B1", statistics.stdev(sam), statistics.mean(ergas), "\u00B1",
                statistics.stdev(ergas)))
        print("psnr: %.2f%s%.2f, ssim: %.2f%s%.2f" % (
                statistics.mean(psnr), "\u00B1", statistics.stdev(psnr), statistics.mean(ssim), "\u00B1",
                statistics.stdev(ssim)))


###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == '__main__':

    file_path = "../data/new_tst(WV3)_0064-0064.h5"
    test(file_path)
