import torch.nn.modules as nn
import torch
import numpy as np
from models import PyramidInjection
import h5py
import scipy.io as sio
import os
import statistics
from PIL import Image
import scipy.io as sio
from util import *
import time
from DatasetProc import load_myset, load_myset_FS

sensor = 'WV3'
ms_channels = 8 if sensor == "WV3" else 4
pix_max_val = 1023. if sensor == "GF2" else 2047.
showingIndices = [4, 2, 1] if sensor == "WV3" else [2, 1, 0]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

ckpt = "pretrained/best%s(mode3).pth" % (sensor)

def test(file_path):
    lms, pan, ums = load_myset_FS(file_path)
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

        ums = ums.astype(np.float32) / pix_max_val
        ums = torch.from_numpy(ums)  # NxCxHxW:

        lms = lms.to(device).float()  # convert to tensor type:
        pan = pan.to(device).float()  # convert to tensor type:
        ums = ums.to(device).float()  # convert to tensor type:

        num_exm = lms.shape[0]

        for index in range(num_exm):  # save the DL results to the 03-Comparisons(Matlab)
            pan_ = pan[index]
            ums_ = ums[index]

            pan_ = pan_.unsqueeze(0)
            ums_ = ums_.unsqueeze(0)
            start = time.time()
            sr = model(ums_, pan_)  # tensor type: sr = NxCxHxW
            end = time.time()
            print("Testing [%d] success,Testing time is [%f]" % (index, end - start))
            ### save result to mat file.
            if 0:
                sr_nd = sr.permute(0, 2, 3, 1).cpu().detach().numpy()  # to: NxHxWxC
                file_name = str(index) + ".mat"
                file_name2 = os.getcwd() + "/MyNet_FS_results_large({})".format(sensor)
                # file_name2 = os.getcwd() + "/results({})".format('NBU')
                mkdir(file_name2)
                # file_name2 = file_name2 + "/mode3(SP3)"
                # mkdir(file_name2)
                save_name = os.path.join(file_name2, file_name)
                sio.savemat(save_name, {'ms_est': sr_nd[0, :, :, :] * pix_max_val})
            #### end of saving

    ###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == '__main__':


	file_path = "../data/tst_fs(%s)_0064-0064.h5" % (sensor)
    test(file_path)