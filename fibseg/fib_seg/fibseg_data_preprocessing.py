from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random

#=========================================================================================
# Fiber Bundle segmentation tool - Data preprocessing
# Vaanathi Sundaresan
# 10-08-2023
#=========================================================================================


def preprocess_data_gauss(data, brain_mask):
    """
    Gaussian intensity normalisation of data
    :param data: input data
    :return: Gaussian normalised data
    """
    brain = brain_mask > 0
    data = data - np.mean(data[brain])
    den = np.std(data[brain])
    if den == 0:
        den = 1
    data = data/den
    data[brain == 0] = np.min(data)
    return data


def minmaxnorm(data):
    data = data - np.min(data)
    data = data/np.amax(data)
    return data


def preprocess_data_gauss(data):
    """
    Gaussian intensity normalisation of data
    :param data: input data
    :return: Gaussian normalised data
    """
    brain1 = data > 0
    brain = brain1 > 0
    data = data - np.mean(data[brain])
    den = np.std(data[brain])
    if den == 0:
        den = 1
    data = data/den
    data[brain == 0] = np.min(data)
    return data


def cut_zeros1d(im_array):
    """
    Find the window for cropping the data closer to the brain
    :param im_array: input array
    :return: starting and end indices, and length of non-zero intensity values
    """
    im_list = list(im_array > 0)
    start_index = im_list.index(1)
    end_index = im_list[::-1].index(1)
    length = len(im_array[start_index:])-end_index
    return start_index, end_index, length


def tight_crop_data(img_data):
    """
    Crop the data tighter to the brain
    :param img_data: input array
    :return: cropped image and the bounding box coordinates and dimensions.
    """
    row_sum = np.sum(np.sum(img_data, axis=1), axis=1)
    col_sum = np.sum(np.sum(img_data, axis=0), axis=1)
    stack_sum = np.sum(np.sum(img_data, axis=1), axis=0)
    rsid, reid, rlen = cut_zeros1d(row_sum)
    csid, ceid, clen = cut_zeros1d(col_sum)
    ssid, seid, slen = cut_zeros1d(stack_sum)
    return img_data[rsid:rsid+rlen, csid:csid+clen, ssid:ssid+slen], [rsid, rlen, csid, clen, ssid, slen]


def get_trainval_names_inj(data_path, lab_da_path, lab_db_path, injsite_path):
    """
    :param data_path:
    :param lab_da_path:
    :param lab_db_path:
    :param injsite_path:
    :return:
    """
    data_path_train = [data_path, lab_da_path, lab_db_path, injsite_path]
    return data_path_train


def select_train_val_names(data_path, lab_da_path, lab_db_path, val_numbers):
    """
    Select training and validation subjects randomly given th no. of validation subjects
    :param data_path: input filepaths
    :param val_numbers: int, number of validation subjects
    :return:
    """
    val_ids = random.choices(list(np.arange(len(data_path))), k=val_numbers)
    train_ids = np.setdiff1d(np.arange(len(data_path)), val_ids)
    hist_path_train = [data_path[ind] for ind in train_ids]
    lab_da_path_train = [lab_da_path[ind] for ind in train_ids]
    lab_db_path_train = [lab_db_path[ind] for ind in train_ids]

    hist_path_val = [data_path[ind] for ind in val_ids]
    lab_da_path_val = [lab_da_path[ind] for ind in val_ids]
    lab_db_path_val = [lab_db_path[ind] for ind in val_ids]

    data_path_train = [hist_path_train, lab_da_path_train, lab_db_path_train]
    data_path_val = [hist_path_val, lab_da_path_val, lab_db_path_val]
    return data_path_train, data_path_val, val_ids


def select_train_val_names_dataonly(data_path, val_numbers):
    """
    Select training and validation subjects randomly given th no. of validation subjects
    :param data_path: input filepaths
    :param val_numbers: int, number of validation subjects
    :return:
    """
    val_ids = random.choices(list(np.arange(len(data_path))), k=val_numbers)
    train_ids = np.setdiff1d(np.arange(len(data_path)), val_ids)
    hist_path_train = [data_path[ind] for ind in train_ids]

    hist_path_val = [data_path[ind] for ind in val_ids]
    return hist_path_train, hist_path_val, val_ids
