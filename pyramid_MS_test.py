import torch.nn.modules as nn
import torch
import numpy as np
from models import PyramidInjection
import h5py
import scipy.io as sio
import os
import time
import statistics
from PIL import Image
import scipy.io as sio
from util import *
from DatasetProc import load_myset

sensor = 'WV3'
ms_channels = 8 if sensor == "WV3" else 4
pix_max_val = 1023. if sensor == "GF2" else 2047.
showingIndices = [4, 2, 1] if sensor == "WV3" else [2, 1, 0]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

ckpt = "pretrained/bestQB(mode3).pth"
# sam: 1.19±0.34, ergas: 0.83±0.24
# psnr: 45.92±3.65, ssim: 0.98±0.01
ckpt = "pretrained/bestQB(mode0).pth"
# sam: 1.20±0.35, ergas: 0.84±0.24
# psnr: 45.89±3.67, ssim: 0.98±0.01

# ckpt = "pretrained/bestGF2(mode3).pth"
# sam: 1.09±0.41, ergas: 1.11±0.64
# psnr: 40.12±4.34, ssim: 0.96±0.03
# ckpt = "pretrained/bestGF2(sp4).pth"
# sam: 1.18±0.44, ergas: 1.24±0.66
# psnr: 39.14±4.19, ssim: 0.95±0.03
# ckpt = "pretrained/bestGF2(sp2).pth"
# sam: 1.11±0.42, ergas: 1.12±0.62
# psnr: 40.02±4.25, ssim: 0.96±0.03
# ckpt = "pretrained/bestGF2(mode0).pth"
# sam: 1.12±0.42, ergas: 1.15±0.63
# psnr: 39.76±4.21, ssim: 0.96±0.03

# ckpt = "pretrained/bestWV2(mode3).pth"
# sam: 1.32±0.58, ergas: 1.02±0.26
# psnr: 42.36±4.62, ssim: 0.98±0.01
# ckpt = "pretrained/bestWV2(mode0).pth"
# sam: 1.33±0.58, ergas: 1.04±0.28
# psnr: 42.21±4.69, ssim: 0.97±0.01
ckpt = "pretrained/bestWV3(mode3).pth"
# sam: 4.12±1.29, ergas: 2.72±0.81
# psnr: 34.78±3.50, ssim: 0.95±0.04
# ckpt = "pretrained/bestWV3(mode0).pth"
# sam: 4.15±1.30, ergas: 2.75±0.82
# psnr: 34.68±3.49, ssim: 0.95±0.04
# ckpt = "pretrained/bestWV3(sp1).pth"
# sam: 4.18±1.30, ergas: 2.80±0.83
# psnr: 34.52±3.47, ssim: 0.95±0.04
# ckpt = "pretrained/bestWV3(sp2).pth"
# sam: 4.14±1.30, ergas: 2.73±0.82
# psnr: 34.74±3.50, ssim: 0.95±0.04
# ckpt = "pretrained/bestWV3(sp4).pth"
# sam: 4.12±1.30, ergas: 2.74±0.82
# psnr: 34.72±3.51, ssim: 0.95±0.04

def test(file_path):
    lms, pan, gt, ums = load_myset(file_path)
    model = PyramidInjection(ms_channels, 3).to(device).eval()
    weight = torch.load(ckpt)
    model.load_state_dict(weight)
    fo = open("result.txt", "a")
    print("Name of the model: ", ckpt)
    print('%s\t%s' % (str(datetime.datetime.now()), ckpt), file=fo)
    with torch.no_grad():

        lms = lms.astype(np.float32) / pix_max_val
        lms = torch.from_numpy(lms)  # NxCxHxW:

        pan = pan.astype(np.float32) / pix_max_val
        pan = torch.from_numpy(pan)  # NxCxHxW:

        gt = gt.astype(np.float32) / pix_max_val
        gt = torch.from_numpy(gt)  # NxCxHxW:

        ums = ums.astype(np.float32) / pix_max_val
        ums = torch.from_numpy(ums)  # NxCxHxW:

        lms = lms.to(device).float()  # convert to tensor type:
        pan = pan.to(device).float()  # convert to tensor type:
        ums = ums.to(device).float()  # convert to tensor type:

        num_exm = lms.shape[0]

        sam = []
        ergas = []
        psnr = []
        ssim = []
        for index in range(num_exm):  # save the DL results to the 03-Comparisons(Matlab)
            pan_ = pan[index]
            ums_ = ums[index]

            pan_ = pan_.unsqueeze(0)
            ums_ = ums_.unsqueeze(0)

            start = time.time()
            sr = model(ums_, pan_)  # tensor type: sr = NxCxHxW
            end = time.time()
            
            ### print testing time
            print("Testing [%d] time: [%f]" % (index, end - start))
            ### save result to mat file.
            if 0:
                sr_nd = sr.permute(0, 2, 3, 1).cpu().detach().numpy()  # to: NxHxWxC
                file_name = str(index) + ".mat"
                file_name2 = os.getcwd() + "/results({})".format(sensor)
                mkdir(file_name2)
                file_name2 = file_name2 + "/mode3(SP3)"
                mkdir(file_name2)
                save_name = os.path.join(file_name2, file_name)
                sio.savemat(save_name, {'ms_est': sr_nd[0, :, :, :] * pix_max_val})
            #### end of saving

            ### calculate some metrics and storing
            psnr.append(calculate_psnr(sr.squeeze().permute(1, 2, 0).cpu().numpy(),
                                       gt[index, :, :, :].permute(1, 2, 0).cpu().numpy(), dynamic_range=1))
            ssim.append(calculate_ssim(sr.squeeze().permute(1, 2, 0).cpu().numpy(),
                                       gt[index, :, :, :].permute(1, 2, 0).cpu().numpy(), dynamic_range=1))
            ergas.append(calculate_ergas(sr.squeeze().permute(1, 2, 0).cpu().numpy(),
                                         gt[index, :, :, :].permute(1, 2, 0).cpu().numpy()))
            sam.append(calculate_sam(sr.squeeze().permute(1, 2, 0).cpu().numpy(),
                                     gt[index, :, :, :].permute(1, 2, 0).cpu().numpy()) * 180 / math.pi)
            #### end of metrics

            ### save to file
            print("No. %d: PSNR:\t%.3f, SSIM:\t%.3f, ERGAS:\t%.3f, SAM:\t%.3f" % (index, psnr[-1], ssim[-1], ergas[-1], sam[-1]), file=fo)

        ### print statistics of metrics
        print("sam: %.2f%s%.2f, ergas: %.2f%s%.2f" % (statistics.mean(sam), "\u00B1", statistics.stdev(sam), statistics.mean(ergas), "\u00B1", statistics.stdev(ergas)))
        print("psnr: %.2f%s%.2f, ssim: %.2f%s%.2f" % (statistics.mean(psnr), "\u00B1", statistics.stdev(psnr), statistics.mean(ssim), "\u00B1", statistics.stdev(ssim)))

    fo.close()
    ###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == '__main__':

    # test("../data/_tst(WV3)_0064-0064.h5")
    # test("../data/_tst(WV2)_0064-0064.h5")
    # test("../data/_tst(QB)_0064-0064_0.h5")
    # test("../data/new_tst(QB)_0064-0064.h5")
    # test("../data/new_tst(WV2)_0064-0064.h5")
    # test("../data/new_tst(WV2)_0064-0064.h5")
    test("../data/new_tst(WV3)_0064-0064.h5")
    # test("../data/large_tst(WV3)_0128-0128.h5")
    # test("../data/validation1.mat")
