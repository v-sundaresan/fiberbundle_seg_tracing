from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fibseg.fib_seg import fibseg_data_preparation, fibseg_data_preprocessing
from fibseg.utils import fibseg_dataset_utils

# =========================================================================================
# Fiber bundle segmentation evaluation function
# Vaanathi Sundaresan
# 10-08-2023
# =========================================================================================


def dice_coeff(inp, tar):
    '''
    Calculating Dice similarity coefficient
    :param inp: Input tensor
    :param tar: Target tensor
    :return: Dice value (scalar)
    '''
    smooth = 1.
    pred_vect = inp.contiguous().view(-1)
    target_vect = tar.contiguous().view(-1)
    intersection = (pred_vect * target_vect).sum()
    dice = (2. * intersection + smooth) / (torch.sum(pred_vect) + torch.sum(target_vect) + smooth)
    return dice


def evaluate_fibseg(test_slice, model, test_params, device, modtype='unet', p_size=256, verbose=False):
    '''
    Truenet evaluate function definition
    :param test_name_dicts: list of dictionaries with test filepaths
    :param model: test model
    :param test_params: parameters used for testing
    :param device: cpu or gpu
    :param mode: acquisition plane
    :param verbose: display debug messages
    :return: predicted probability array
    '''
    numchannels = test_params['Num_channels']
    testdata = fibseg_data_preparation.load_and_crop_test_julia_patches(test_slice, patchsize=p_size,
                                                                            use_reduce=False)
    testdata = testdata / np.amax(testdata)
    # testdata = preprocess_data_gauss(testdata)
    if verbose:
        print('Testdata dimensions.......................................')
        print(testdata.shape)

    if numchannels != testdata.shape[1]:
        raise ValueError('Number of input channels in the model do not match with the number of data channels')

    test_dataset_dict = fibseg_dataset_utils.HistTestDataset(testdata)
    test_dataloader = DataLoader(test_dataset_dict, batch_size=1, shuffle=False, num_workers=0)

    model.eval()
    segout1 = np.array([])
    reconout = np.array([])
    svxout = np.array([])
    prob_array = np.array([])
    with torch.no_grad():
        for _, test_dict in enumerate(test_dataloader):
            X = test_dict['hist']
            X = X.to(device=device, dtype=torch.float32)
            print(svx.size())
            if modtype == 'transunet':
                syseg, _ = model(X)
            elif modtype == 'unet':
                syseg, syrecon = model(X, 64)
                syrecon = syrecon.cpu().detach().numpy()
                reconout = np.concatenate([reconout, syrecon], axis=0) if reconout.size else syrecon
            softmax = nn.Softmax()
            syseg = softmax(syseg)
            svx = svx.cpu().detach().numpy()
            syseg = syseg.cpu().detach().numpy()
            segout1 = np.concatenate([segout1, syseg], axis=0) if segout1.size else syseg
            svxout = np.concatenate([svxout, svx], axis=0) if svxout.size else svx

    if modtype == 'unet':
        return segout1, svxout, reconout
    else:
        return fibseg_data_preprocessing.minmaxnorm(np.exp(segout1)), svxout


