# Utilities for testing model
# Vaanathi Sundaresan, 2022
# arXiv:2208.03569
########################################################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import util_fns.data_preparation_utils as data_preparation_utils
from util_fns.utils import *
from util_fns.dataset_utils import HistTestDataset
from model.model_utils import *
from torch.utils.data import DataLoader, Dataset
########################################################################################################


def test_histseg_temp_ensembling(source_test_data, model, batch_size, device, p_size=256):
    model.eval()
    testdata = data_preparation_utils.load_and_crop_test_julia_patches_omni(source_test_data, patchsize=p_size)
    testdata = testdata / np.amax(testdata)
    # testdata = preprocess_data_gauss(testdata)
    test_dataset_dict = HistTestDataset(testdata)
    test_dataloader = DataLoader(test_dataset_dict, batch_size=batch_size, shuffle=False, num_workers=0)

    segout = np.array([])
    reconout = np.array([])
    svxout = np.array([])
    for _, test_dict in enumerate(test_dataloader):
        svx = test_dict['hist']
        svx = svx.to(device=device, dtype=torch.float32)
        print(svx.size())
        syseg, syclass = model(svx, 64)
        syclass = syclass.cpu().detach().numpy()
        reconout = np.concatenate([reconout, syclass], axis=0) if reconout.size else syclass
        softmax = nn.Softmax()
        syseg = softmax(syseg)
        svx = svx.cpu().detach().numpy()
        syseg = syseg.cpu().detach().numpy()
        segout = np.concatenate([segout, syseg], axis=0) if segout.size else syseg
        svxout = np.concatenate([svxout, svx], axis=0) if svxout.size else svx

    return segout, svxout, reconout










