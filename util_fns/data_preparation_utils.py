from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import nibabel as nib
import random
from skimage.transform import resize
from skimage.measure import regionprops, label
import hist_seg_patches_augmentation
from skimage import filters

# =========================================================================================
# Data preparation code for histology data (nifti files)
# Vaanathi Sundaresan
# 12-18-2021, MGH Martinos MA, USA
# =========================================================================================


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


def load_and_align_histdata(slidenames):
    '''
    :param slidenames: Filenames of slides for a single brain samples from diff compartments
    :return: 3-channel slide sequence (N x H x W x C)
    '''
    print(slidenames)
    hdims = []
    wdims = []
    for idx in range(len(slidenames)):
        slide_data = nib.load(slidenames[idx]).get_data().astype('float32')
        print(slide_data.shape)
        hdims.append(slide_data.shape[0])
        wdims.append(slide_data.shape[1])

    print(hdims)
    print(wdims)
    # Get the maximum H and W dimensions to align the slides
    hmax = max(hdims) if max(hdims) % 2 == 0 else max(hdims)+1
    wmax = max(wdims) if max(wdims) % 2 == 0 else max(wdims)+1

    slide_seq = np.zeros([hmax, wmax, 3], dtype='float32')

    # Centering and aligning the slides into a sequence
    for idx in range(len(slidenames)):
        slide_data = nib.load(slidenames[idx]).get_data().astype('float32')
        startrow = hmax // 2 - slide_data.shape[0] // 2
        startcol = wmax // 2 - slide_data.shape[1] // 2
        endrow = startrow + slide_data.shape[0]
        endcol = startcol + slide_data.shape[1]
        slide_seq[startrow:endrow, startcol:endcol, :] = slide_seq[startrow:endrow, startcol:endcol, :] + \
                                                         np.squeeze(slide_data[:, :, 0, :])/np.amax(slide_data)

    return slide_seq


def load_and_crop_training_data_regions(slidenames, labelnames, num_patches=10, patchsize=1024):
    """
    :param slidenames: list(strings) - file paths
    :param labelnames: list(strings) - file paths
    :param num_patches: int - number of randomly sampled patches to generate
    :param patchsize: int (2^n) - dimensions of randomly sampled patches (height=width)
    :return:
    """
    all_patches = np.array([])
    all_lpatches = np.array([])
    factor = 8
    for idx in range(len(slidenames)):
        slide_data = nib.load(slidenames[idx]).get_data().astype('float32')
        slide_data = np.squeeze(slide_data[:,:,0,:])
        slide_data = resize(slide_data, (slide_data.shape[0]//factor, slide_data.shape[1]//factor, slide_data.shape[2]), preserve_range=True)

        label_data3d = nib.load(labelnames[idx]).get_data().astype('float32')
        label_data = np.zeros([label_data3d.shape[0], label_data3d.shape[1]])
        # label_data[label_data3d[:, :, 0, 0] > 0] = 1
        # label_data[label_data3d[:, :, 0, 1] > 0] = 1
        label_data[label_data3d[:, :, 0, 2] > 0] = 1
        [width, height, _] = slide_data.shape
        rowarray = np.arange(width // 6, (width * 5) // 6)
        colarray = np.arange(height // 6, (height * 5) // 6)
        row_ids = np.random.choice(rowarray, num_patches, replace=False)
        col_ids = np.random.choice(colarray, num_patches, replace=False)
        patches = np.zeros([num_patches, patchsize, patchsize, 3])
        lpatches = np.zeros([num_patches, patchsize, patchsize, 1])
        for patch in range(num_patches):
            startrow = np.amax([row_ids[patch] - patchsize // 2, 0])
            startcol = np.amax([col_ids[patch] - patchsize // 2, 0])
            endrow = np.amin([row_ids[patch] + patchsize // 2, width])
            endcol = np.amin([col_ids[patch] + patchsize // 2, height])
            patchdata = slide_data[startrow:endrow, startcol:endcol, :]
            [pwidth, pheight, _] = patchdata.shape
            print(pwidth, pheight)
            patches[patch, :pwidth, :pheight, :] = patchdata
            lpatchdata = label_data[startrow:endrow, startcol:endcol]
            lpatches[patch, :pwidth, :pheight, 0] = lpatchdata
        all_patches = np.concatenate([all_patches, patches], axis=0) if all_patches.size else patches
        all_lpatches = np.concatenate([all_lpatches, lpatches], axis=0) if all_lpatches.size else lpatches
    all_patches = all_patches.transpose(0, 3, 1, 2)
    all_lpatches = all_lpatches.transpose(0, 3, 1, 2)
    return [all_patches, all_lpatches]


def load_and_crop_training_juliadata_regions(slidenames, labelnamesda, labelnamesdb, num_patches=10, patchsize=1024,
                                             augment=False):
    """
    :param slidenames: list(strings) - file paths
    :param labelnames: list(strings) - file paths
    :param num_patches: int - number of randomly sampled patches to generate
    :param patchsize: int (2^n) - dimensions of randomly sampled patches (height=width)
    :return:
    """
    all_patches = np.array([])
    all_lpatches = np.array([])
    factor = 2
    for idx in range(len(slidenames)):
        slide_data = np.load(slidenames[idx]).astype('float32')
        slide_data = resize(slide_data, (slide_data.shape[0]//factor, slide_data.shape[1]//factor, slide_data.shape[2]),
                            preserve_range=True)

        label_data_da = np.load(labelnamesda[idx]).astype('float32')
        label_data_db = np.load(labelnamesdb[idx]).astype('float32')
        label_data = ((label_data_da + label_data_db) > 0).astype('float32')

        label_data = (resize(label_data,
                             (slide_data.shape[0], slide_data.shape[1]),
                             preserve_range=True) > 0).astype('float32')

        [width, height, _] = slide_data.shape
        rowarray = np.arange(width // 6, (width * 5) // 6)
        colarray = np.arange(height // 6, (height * 5) // 6)
        row_ids = np.random.choice(rowarray, num_patches, replace=False)
        col_ids = np.random.choice(colarray, num_patches, replace=False)
        patches = np.zeros([num_patches, patchsize, patchsize, 3])
        lpatches = np.zeros([num_patches, patchsize, patchsize, 1])
        for patch in range(num_patches):
            startrow = np.amax([row_ids[patch] - patchsize // 2, 0])
            startcol = np.amax([col_ids[patch] - patchsize // 2, 0])
            endrow = np.amin([row_ids[patch] + patchsize // 2, width])
            endcol = np.amin([col_ids[patch] + patchsize // 2, height])
            patchdata = slide_data[startrow:endrow, startcol:endcol, :]
            [pwidth, pheight, _] = patchdata.shape
            patches[patch, :pwidth, :pheight, :] = patchdata
            lpatchdata = label_data[startrow:endrow, startcol:endcol]
            lpatches[patch, :pwidth, :pheight, 0] = lpatchdata
        all_patches = np.concatenate([all_patches, patches], axis=0) if all_patches.size else patches
        all_lpatches = np.concatenate([all_lpatches, lpatches], axis=0) if all_lpatches.size else lpatches
        if augment is True:
            all_patches, all_lpatches = perform_augmentation(all_patches, all_lpatches, af=1)
    all_patches = all_patches.transpose(0, 3, 1, 2)
    all_lpatches = all_lpatches.transpose(0, 3, 1, 2)
    return [all_patches, all_lpatches]


def load_and_crop_TEtraining_regions(slidenames, label_data, num_patches=10, patchsize=1024, augment=False):
    """
    :param slidenames: list(strings) - file paths
    :param num_patches: int - number of randomly sampled patches to generate
    :param patchsize: int (2^n) - dimensions of randomly sampled patches (height=width)
    :return:
    """
    all_patches = np.array([])
    all_lpatches = np.array([])
    factor = 4
    for idx in range(len(slidenames)):
        slide_data = np.load(slidenames[idx]).astype('float32')
        slide_data = resize(slide_data, (slide_data.shape[0]//factor, slide_data.shape[1]//factor, slide_data.shape[2]),
                            preserve_range=True)
        label_data = (resize(label_data, (slide_data.shape[0], slide_data.shape[1]),
                             preserve_range=True) > 0).astype('float32')

        [width, height, _] = slide_data.shape
        slide_datach1 = slide_data[:, :, 0]
        slide_datach2 = slide_data[:, :, 1]
        slide_datach3 = slide_data[:, :, 2]
        slide_datachm1 = filters.median(slide_datach1, np.ones((7, 7)))
        slide_datachm2 = filters.median(slide_datach2, np.ones((7, 7)))
        slide_datachm3 = filters.median(slide_datach3, np.ones((7, 7)))
        slide_data = np.concatenate([np.expand_dims(slide_datachm1, axis=-1), np.expand_dims(slide_datachm2, axis=-1)])
        slide_data = np.concatenate([slide_data, np.expand_dims(slide_datachm3, axis=-1)])

        rowarray = np.arange(width // 6, (width * 5) // 6)
        colarray = np.arange(height // 6, (height * 5) // 6)
        row_ids = np.random.choice(rowarray, num_patches, replace=False)
        col_ids = np.random.choice(colarray, num_patches, replace=False)
        patches = np.zeros([num_patches, patchsize, patchsize, 3])
        lpatches = np.zeros([num_patches, patchsize, patchsize, 1])
        for patch in range(num_patches):
            startrow = np.amax([row_ids[patch] - patchsize // 2, 0])
            startcol = np.amax([col_ids[patch] - patchsize // 2, 0])
            endrow = np.amin([row_ids[patch] + patchsize // 2, width])
            endcol = np.amin([col_ids[patch] + patchsize // 2, height])
            patchdata = slide_data[startrow:endrow, startcol:endcol, :]
            [pwidth, pheight, _] = patchdata.shape
            patches[patch, :pwidth, :pheight, :] = patchdata
            lpatchdata = label_data[startrow:endrow, startcol:endcol]
            lpatches[patch, :pwidth, :pheight, 0] = lpatchdata
        all_patches = np.concatenate([all_patches, patches], axis=0) if all_patches.size else patches
        all_lpatches = np.concatenate([all_lpatches, lpatches], axis=0) if all_lpatches.size else lpatches
        if augment is True:
            all_patches, _ = perform_augmentation(all_patches, all_lpatches, af=1)
    all_patches = all_patches.transpose(0, 3, 1, 2)
    all_lpatches = all_lpatches.transpose(0, 3, 1, 2)
    return [all_patches, all_lpatches]


def perform_augmentation(otr, otr_labs, af=2):
    """
    :param otr:
    :param otr_labs:
    :param af:
    :return:
    """
    augmented_img_list = []
    augmented_mseg_list = []
    for i in range(0, af):
        for id in range(otr.shape[0]):
            image = otr[id, :, :, :]
            manmask = otr_labs[id, :, :, :]
            augmented_img, augmented_manseg = hist_seg_patches_augmentation.augment1(image, manmask)
            augmented_img_list.append(augmented_img)
            augmented_mseg_list.append(augmented_manseg)
    augmented_img = np.array(augmented_img_list)
    augmented_mseg = np.array(augmented_mseg_list)
    augmented_img = np.reshape(augmented_img, [-1, otr.shape[1], otr.shape[2], 3])
    augmented_mseg = np.reshape(augmented_mseg, [-1, otr.shape[1], otr.shape[2], 1])
    augmented_imgs = np.tile(augmented_img, (1, 1, 1, 1))
    augmented_mseg = np.tile(augmented_mseg, (1, 1, 1, 1))
    otr_aug = np.concatenate((otr, augmented_imgs), axis=0)
    otr_labs = np.concatenate((otr_labs, augmented_mseg), axis=0)
    return otr_aug, otr_labs


def load_and_crop_test_data_slices(slidenames):  # , labelnames):
    """
    :param slidenames: list(strings) - file paths
    """
    all_slices = np.array([])
    all_labels = np.array([])
    factor = 8
    for idx in range(len(slidenames)):
        slide_data = nib.load(slidenames[idx]).get_data().astype('float32')
        slide_data = np.squeeze(slide_data[:,:,0,:])
        slide_data = resize(slide_data, (slide_data.shape[0]//factor, slide_data.shape[1]//factor, slide_data.shape[2]), preserve_range=True)
        # slide_data = ndimage.gaussian_filter(slide_data, sigma=(11, 11, 0), order=0)
        print(slide_data.shape)
        # label_data3d = nib.load(labelnames[idx]).get_data().astype('float32')
        # label_data = np.zeros([label_data3d.shape[0], label_data3d.shape[1]])
        # label_data[label_data3d[:, :, 0, 0] > 0] = 1
        # label_data[label_data3d[:, :, 0, 1] > 0] = 2
        # label_data[label_data3d[:, :, 0, 2] > 0] = 3
        slides = np.tile(slide_data, [1, 1, 1, 1])
        slides = slides.transpose(0, 3, 1, 2)
        # labels = np.tile(label_data, [1, 1, 1])
        all_slices = np.concatenate([all_slices, slides], axis=0) if all_slices.size else slides
        # all_labels = np.concatenate([all_labels, labels], axis=0) if all_labels.size else labels

    return all_slices


def load_and_crop_test_juliadata_slices(slidenames):  # , labelnames):
    """
    :param slidenames: list(strings) - file paths
    """
    all_slices = np.array([])
    all_labels = np.array([])
    for idx in range(len(slidenames)):
        slide_data = np.load(slidenames[idx]).astype('float32')
        slide_data = np.squeeze(slide_data)
        if slide_data.shape[1] > 4000:
            factor = 4
        else:
            factor = 2
        slide_data = resize(slide_data, (slide_data.shape[0]//factor, slide_data.shape[1]//factor, slide_data.shape[2]), preserve_range=True)
        # slide_data = ndimage.gaussian_filter(slide_data, sigma=(11, 11, 0), order=0)
        print(slide_data.shape)
        # label_data3d = nib.load(labelnames[idx]).get_data().astype('float32')
        # label_data = np.zeros([label_data3d.shape[0], label_data3d.shape[1]])
        # label_data[label_data3d[:, :, 0, 0] > 0] = 1
        # label_data[label_data3d[:, :, 0, 1] > 0] = 2
        # label_data[label_data3d[:, :, 0, 2] > 0] = 3
        slides = np.tile(slide_data, [1, 1, 1, 1])
        slides = slides.transpose(0, 3, 1, 2)
        # labels = np.tile(label_data, [1, 1, 1])
        all_slices = np.concatenate([all_slices, slides], axis=0) if all_slices.size else slides
        # all_labels = np.concatenate([all_labels, labels], axis=0) if all_labels.size else labels

    return all_slices


def plot_prior_patches(coords, pmask, radius=100):
    coords = np.reshape(coords, (-1, 2))
    for i in range(coords.shape[0]):
        start_row = np.amax([np.round(coords[i, 0]).astype(int) - radius, 0])
        start_col = np.amax([np.round(coords[i, 1]).astype(int) - radius, 0])
        end_row = np.amin([np.round(coords[i, 0]).astype(int) + radius, pmask.shape[0]])
        end_col = np.amin([np.round(coords[i, 1]).astype(int) + radius, pmask.shape[1]])
        pmask[start_row:end_row, start_col:end_col] = 1
    return pmask


def get_training_juliapatches_withprior(slide_data, consec_slide_coords, labelnamesda, labelnamesdb, priormap,
                                        num_patches=10, patchsize=1024):

    label_data_da = np.load(labelnamesda).astype('float32')
    label_data_db = np.load(labelnamesdb).astype('float32')
    label_data = ((label_data_da + label_data_db) > 0).astype('float32')

    label_data = (resize(label_data,
                         (slide_data.shape[0], slide_data.shape[1]),
                         preserve_range=True) > 0).astype('float32')

    if priormap is None:
        priormap = np.zeros_like(slide_data)
    else:
        priormap = resize(priormap, (slide_data.shape[0], slide_data.shape[1]), preserve_range=True)

    # Consec_slide_coords has coords for rostral and caudal slides for both prediction and injection sites
    priormap = plot_prior_patches(consec_slide_coords, priormap, radius=250)

    [width, height, _] = slide_data.shape
    rowarray = np.arange(width // 6, (width * 5) // 6)
    colarray = np.arange(height // 6, (height * 5) // 6)
    row_ids = np.random.choice(rowarray, num_patches, replace=False)
    col_ids = np.random.choice(colarray, num_patches, replace=False)
    patches = np.zeros([num_patches, patchsize, patchsize, 3])
    lpatches = np.zeros([num_patches, patchsize, patchsize, 1])
    prpatches = np.zeros([num_patches, patchsize, patchsize, 1])
    for patch in range(num_patches):
        startrow = np.amax([row_ids[patch] - patchsize // 2, 0])
        startcol = np.amax([col_ids[patch] - patchsize // 2, 0])
        endrow = np.amin([row_ids[patch] + patchsize // 2, width])
        endcol = np.amin([col_ids[patch] + patchsize // 2, height])
        patchdata = slide_data[startrow:endrow, startcol:endcol, :]
        [pwidth, pheight, _] = patchdata.shape
        patches[patch, :pwidth, :pheight, :] = patchdata
        lpatchdata = label_data[startrow:endrow, startcol:endcol]
        lpatches[patch, :pwidth, :pheight, 0] = lpatchdata
        prpatchdata = label_data[startrow:endrow, startcol:endcol]
        prpatches[patch, :pwidth, :pheight, 0] = prpatchdata
    patches = patches.transpose(0, 3, 1, 2)
    lpatches = lpatches.transpose(0, 3, 1, 2)
    prpatches = prpatches.transpose(0, 3, 1, 2)
    traindata = [patches, lpatches, prpatches]

    num_val_patches = num_patches // 3
    row_ids = np.random.choice(rowarray, num_val_patches, replace=False)
    col_ids = np.random.choice(colarray, num_val_patches, replace=False)
    patches = np.zeros([num_val_patches, patchsize, patchsize, 3])
    lpatches = np.zeros([num_val_patches, patchsize, patchsize, 1])
    prpatches = np.zeros([num_val_patches, patchsize, patchsize, 1])
    for patch in range(num_val_patches):
        startrow = np.amax([row_ids[patch] - patchsize // 2, 0])
        startcol = np.amax([col_ids[patch] - patchsize // 2, 0])
        endrow = np.amin([row_ids[patch] + patchsize // 2, width])
        endcol = np.amin([col_ids[patch] + patchsize // 2, height])
        patchdata = slide_data[startrow:endrow, startcol:endcol, :]
        [pwidth, pheight, _] = patchdata.shape
        patches[patch, :pwidth, :pheight, :] = patchdata
        lpatchdata = label_data[startrow:endrow, startcol:endcol]
        lpatches[patch, :pwidth, :pheight, 0] = lpatchdata
        prpatchdata = label_data[startrow:endrow, startcol:endcol]
        prpatches[patch, :pwidth, :pheight, 0] = prpatchdata
    patches = patches.transpose(0, 3, 1, 2)
    lpatches = lpatches.transpose(0, 3, 1, 2)
    prpatches = prpatches.transpose(0, 3, 1, 2)
    valdata = [patches, lpatches, prpatches]

    return traindata, valdata, priormap


def load_test_julia_slide(slidename):  # , labelnames):
    """
    :param slidename: strings - file paths
    """

    slide_data = np.load(slidename).astype('float32')
    slide_data = np.squeeze(slide_data)
    if slide_data.shape[1] > 4000:
        factor = 4
    else:
        factor = 2
    slide_data = resize(slide_data, (slide_data.shape[0]//factor, slide_data.shape[1]//factor, slide_data.shape[2]), preserve_range=True)
    print(slide_data.shape)
    slides = np.tile(slide_data, [1, 1, 1, 1])
    slides = slides.transpose(0, 3, 1, 2)

    return slides


def load_and_crop_test_julia_patches(slidename, patchsize=256):
    factor = 2
    slide_data = np.load(slidename).astype('float32')
    # slide_data = resize(slide_data,
    #                     (slide_data.shape[0] // factor, slide_data.shape[1] // factor, slide_data.shape[2]),
    #                     preserve_range=True)
    slide_data = np.squeeze(slide_data)
    height, width, _ = slide_data.shape
    num_along_height = np.ceil(height / patchsize).astype(int)
    num_along_width = np.ceil(width / patchsize).astype(int)
    num_patches = np.ceil(num_along_height * num_along_width).astype(int)
    test_patches = np.zeros([num_patches, patchsize, patchsize, slide_data.shape[2]])
    count = 0
    for row in range(num_along_height):
        for col in range(num_along_width):
            start_row = np.amax([row * patchsize, 0])
            end_row = np.amin([(row + 1) * patchsize, slide_data.shape[0]])
            start_col = np.amax([col * patchsize, 0])
            end_col = np.amin([(col + 1) * patchsize, slide_data.shape[1]])
            tmp_patch = slide_data[start_row:end_row, start_col:end_col, :]
            tmph, tmpw, _ = tmp_patch.shape
            print('###############PATCHSIZE DEBUGGING###################', flush=True)
            print(tmp_patch.shape, flush=True)
            # print(test_patches.shape)
            # print(tmph, tmpw)
            test_patches[count, :tmph, :tmpw, :] = tmp_patch
            count += 1
    test_patches = test_patches.transpose(0, 3, 1, 2)

    return test_patches


def load_and_crop_test_julia_patches_TE(slidename, patchsize=256, use_reduce=True, factor=2):
    slide_data = np.load(slidename).astype('float32')
    if use_reduce is True:
        slide_data = resize(slide_data,
                            (slide_data.shape[0] // factor, slide_data.shape[1] // factor, slide_data.shape[2]),
                            preserve_range=True)
    slide_data = np.squeeze(slide_data)
    height, width, _ = slide_data.shape
    # print(slide_data.shape)
    # slide_datach1 = slide_data[:, :, 0]
    # slide_datach2 = slide_data[:, :, 1]
    # slide_datach3 = slide_data[:, :, 2]
    # # print(slide_datach1.shape)
    # # print(slide_datach2.shape)
    # # print(slide_datach3.shape)
    # slide_datachm1 = filters.median(slide_datach1, np.ones((7, 7)))
    # slide_datachm2 = filters.median(slide_datach2, np.ones((7, 7)))
    # slide_datachm3 = filters.median(slide_datach3, np.ones((7, 7)))
    # # print(slide_datachm1.shape)
    # # print(slide_datachm2.shape)
    # # print(slide_datachm3.shape)
    # slide_data = np.concatenate([np.expand_dims(slide_datachm1, axis=-1), np.expand_dims(slide_datachm2, axis=-1)],
    #                             axis=-1)
    # # print(slide_data.shape)
    # slide_data = np.concatenate([slide_data, np.expand_dims(slide_datachm3, axis=-1)], axis=-1)
    print(slide_data.shape)
    num_along_height = np.ceil(height / patchsize).astype(int)
    num_along_width = np.ceil(width / patchsize).astype(int)
    num_patches = np.ceil(num_along_height * num_along_width).astype(int)
    test_patches = np.zeros([num_patches, patchsize, patchsize, slide_data.shape[2]])
    count = 0
    for row in range(num_along_height):
        for col in range(num_along_width):
            start_row = np.amax([row * patchsize, 0])
            end_row = np.amin([(row + 1) * patchsize, slide_data.shape[0]])
            start_col = np.amax([col * patchsize, 0])
            end_col = np.amin([(col + 1) * patchsize, slide_data.shape[1]])
            tmp_patch = slide_data[start_row:end_row, start_col:end_col, :]
            tmph, tmpw, _ = tmp_patch.shape
            print('###############PATCHSIZE DEBUGGING###################', flush=True)
            print(tmp_patch.shape, flush=True)
            # print(test_patches.shape)
            # print(tmph, tmpw)
            test_patches[count, :tmph, :tmpw, :] = tmp_patch
            count += 1
    test_patches = test_patches.transpose(0, 3, 1, 2)

    return test_patches


def load_dsdatavols_for_priormaptraining(train_hist_names, train_labels, num_cases=16):
    histdata = np.load(train_hist_names)
    labdata = np.load(train_labels)
    start_col = histdata.shape[2] // 2 - 128
    end_col = histdata.shape[2] // 2 + 128

    histdata = resize(histdata[:, :, start_col:end_col, :], (histdata.shape[0], 128, 128, 3))
    labdata = resize(labdata[:, :, start_col:end_col], (labdata.shape[0], 128, 128))
    slide_ids = random.sample(list(np.arange(4, 19)), num_cases)

    hist_instances = []
    lab_instances = []

    for ind in range(len(slide_ids)):
        start_stack = np.max([ind-8, 0])
        end_stack = np.min([ind+8, histdata.shape[0]])
        hist_instance = np.zeros([16, 128, 128, 3])
        lab_instance = np.zeros([16, 128, 128])
        hist_stack = histdata[start_stack:end_stack, :, :, :]
        lab_stack = labdata[start_stack:end_stack, :, :]
        print(lab_stack.shape)
        hist_instance[:hist_stack.shape[0], :, :, :] = hist_stack
        lab_instance[:lab_stack.shape[0], :, :] = lab_stack
        hist_instances.append(hist_instance)
        lab_instances.append(lab_instance)

    hist_instances = np.array(hist_instances)
    hist_instances = np.reshape(hist_instances, [-1, 16, 128, 128, 3])
    hist_instances = hist_instances.transpose(0, 4, 2, 3, 1)

    lab_instances = np.array(lab_instances)
    lab_instances = np.reshape(lab_instances, [-1, 16, 128, 128])
    lab_instances = lab_instances.transpose(0, 2, 3, 1)

    return hist_instances, lab_instances


def load_data_for_stackdet_finetune(slidenames, labelnamesda, labelnamesdb, num_patches=10, patchsize=1024,
                                    augment=False):
    all_patches = np.array([])
    all_lpatches = np.array([])
    factor = 2
    for idx in range(len(slidenames)):
        slide_data = np.load(slidenames[idx]).astype('float32')
        slide_data = resize(slide_data,
                            (slide_data.shape[0] // factor, slide_data.shape[1] // factor, slide_data.shape[2]),
                            preserve_range=True)

        label_data_da = np.load(labelnamesda[idx]).astype('float32')
        label_data_db = np.load(labelnamesdb[idx]).astype('float32')
        label_data = ((label_data_da + label_data_db) > 0).astype('float32')

        try:
            extra_label_data_da = np.load(labelnamesda[idx][:-27] + 'Fiber_bundle_dense_mask_extra.npy').astype(float)
        except:
            extra_label_data_da = np.zeros_like(label_data)

        label_data = ((label_data + extra_label_data_da) > 0).astype(float)

        label_data = (resize(label_data,
                             (slide_data.shape[0], slide_data.shape[1]),
                             preserve_range=True) > 0).astype('float32')

        [width, height, _] = slide_data.shape
        if slidenames[idx][-23:-16] == 'mr256LY':
            print('yes!')
            props = regionprops(label(label_data > 0))
            bbox = props[0].bbox  # (min_row, min_col, max_row, max_col)
            rowarray = np.arange(bbox[0], bbox[2])
            colarray = np.arange(bbox[1], bbox[3])
            print(rowarray)
            print(colarray)
        else:
            rowarray = np.arange(width // 6, (width * 5) // 6)
            colarray = np.arange(height // 6, (height * 5) // 6)
        try:
            row_ids = np.random.choice(rowarray, num_patches, replace=False)
            col_ids = np.random.choice(colarray, num_patches, replace=False)
        except:
            row_ids = np.random.choice(rowarray, num_patches, replace=True)
            col_ids = np.random.choice(colarray, num_patches, replace=True)
        patches = np.zeros([num_patches, patchsize, patchsize, 3])
        lpatches = np.zeros([num_patches, patchsize, patchsize, 1])
        for patch in range(num_patches):
            startrow = np.amax([row_ids[patch] - patchsize // 2, 0])
            startcol = np.amax([col_ids[patch] - patchsize // 2, 0])
            endrow = np.amin([row_ids[patch] + patchsize // 2, width])
            endcol = np.amin([col_ids[patch] + patchsize // 2, height])
            patchdata = slide_data[startrow:endrow, startcol:endcol, :]
            [pwidth, pheight, _] = patchdata.shape
            patches[patch, :pwidth, :pheight, :] = patchdata
            lpatchdata = label_data[startrow:endrow, startcol:endcol]
            lpatches[patch, :pwidth, :pheight, 0] = lpatchdata
        all_patches = np.concatenate([all_patches, patches], axis=0) if all_patches.size else patches
        all_lpatches = np.concatenate([all_lpatches, lpatches], axis=0) if all_lpatches.size else lpatches
        if augment is True:
            all_patches, all_lpatches = perform_augmentation(all_patches, all_lpatches, af=1)
    all_patches = all_patches.transpose(0, 3, 1, 2)
    all_lpatches = all_lpatches.transpose(0, 3, 1, 2)
    return [all_patches, all_lpatches]







