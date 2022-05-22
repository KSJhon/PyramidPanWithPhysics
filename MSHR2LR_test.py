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

ckpt = "pretrained/hr2lr%s.pth" % (sensor)
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

            if 1:
                sr_nd = res.permute(0, 2, 3, 1).cpu().detach().numpy()  # to: NxHxWxC
                file_name1 = os.getcwd() + "/H2L_results({})".format(sensor)


                mkdir(file_name1)

                save_name1 = os.path.join(file_name1, file_name)
                sio.savemat(save_name1, {'m2l_est': sr_nd[0, :, :, :] * pix_max_val})
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

	file_path = "../data/tst(%s)_0064-0064.h5" % (sensor)
    test(file_path)
