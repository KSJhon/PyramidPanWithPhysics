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

ckpt = "pretrained/best%s(mode3).pth" % (sensor)

def test(file_path):
    lms, pan, gt, ums = load_myset(file_path)
    model = PyramidInjection(ms_channels, 3).to(device).eval()
    weight = torch.load(ckpt)
    model.load_state_dict(weight, device)
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
		times = []
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
                file_path = os.getcwd() + "/results({})".format(sensor)
                mkdir(file_path)
                file_path = file_path + "/mode3(SP3)"
                mkdir(file_path)
                save_name = os.path.join(file_path, file_name)
                sio.savemat(save_name, {'ms_est': sr_nd[0, :, :, :] * pix_max_val})
            #### end of saving

            ### calculate some metrics and storing
			times.append((end - start) * 1000)
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
		
		times.pop(0)
        print("avg. running time:%.3f" % (statistics.mean(times)))
    fo.close()
    ###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == '__main__':
    file_path = "../data/tst(%s)_0064-0064.h5" % (sensor)
    test(file_path)

