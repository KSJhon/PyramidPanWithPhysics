import torch.utils.data as data
import torch
import h5py
import cv2
import numpy as np
import scipy.io as sio
from util import get_edge1

class DatasetProc4Physic(data.Dataset):
    def __init__(self, file_path, max_val=2047.0):
        super(DatasetProc4Physic, self).__init__()
        data = h5py.File(file_path, mode='r')  # NxCxHxW = 0x1x2x3

        # tensor type:

        pans_org = data["PANS"][...].astype(np.float32) / max_val  # convert to np tpye for CV2.filter
        unpermuted = torch.from_numpy(pans_org)    #for mat file version 7.3
        self.pans = unpermuted.permute(3, 2, 0, 1)

        lm_org = data["LRMSS"][...].astype(np.float32) / max_val  # convert to np tpye for CV2.filter
        unpermuted = torch.from_numpy(lm_org)
        self.lrmss = unpermuted.permute(3, 2, 0, 1)

        hm_org = data["HRMSS"][...].astype(np.float32) / max_val  # convert to np tpye for CV2.filter
        unpermuted = torch.from_numpy(hm_org)
        self.hrmss = unpermuted.permute(3, 2, 0, 1)

        us_org = data["USMSS"][...].astype(np.float32) / max_val  # convert to np tpye for CV2.filter
        unpermuted = torch.from_numpy(us_org)
        self.usmss = unpermuted.permute(3, 2, 0, 1)

        print(self.usmss.size())
        print(self.usmss[100, 1, 1, 1])

    def __getitem__(self, index):
        return self.pans[index, :, :, :].float(), \
               self.lrmss[index, :, :, :].float(), \
               self.hrmss[index, :, :, :].float(), \
               self.usmss[index, :, :, :].float(),

    def __len__(self):
        return self.pans.shape[0]

def load_myset(file_path):
    ## ===== case1: NxCxHxW
    data = h5py.File(file_path, mode='r')
    dataset_names = list(data.keys())
    lms = data["LRMSS"][...]  # NxCxHxW=0,1,2,3
    lms = lms.transpose(3, 2, 0, 1)
    shape_size = len(lms.shape)

    gt = data["HRMSS"][...]
    gt = gt.transpose(3, 2, 0, 1)

    pan = data["PANS"][...]
    pan = pan.transpose(3, 2, 0, 1)

    ums = data["USMSS"][...]
    ums = ums.transpose(3, 2, 0, 1)

    assert (shape_size == 4)

    return lms, pan, gt, ums


def load_myset_FS(file_path):
    ## ===== case1: NxCxHxW
    data = h5py.File(file_path, mode='r')
    dataset_names = list(data.keys())
    lms = data["LRMSS"][...]  # NxCxHxW=0,1,2,3
    lms = lms.transpose(3, 2, 0, 1)
    shape_size = len(lms.shape)

    pan = data["PANS"][...]
    pan = pan.transpose(3, 2, 0, 1)

    ums = data["USMSS"][...]
    ums = ums.transpose(3, 2, 0, 1)

    assert (shape_size == 4)

    return lms, pan, ums

