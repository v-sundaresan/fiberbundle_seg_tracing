from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from skimage.transform import resize
from fibseg.fib_seg import fibseg_data_augmentation

#=========================================================================================
# Fiber Bundle segmentation tool - Data preparation code
# Vaanathi Sundaresan
# 10-08-2023
#=========================================================================================


def load_and_crop_test_julia_patches_te(slide_data, patchsize=256, use_reduce=True, factor=2):
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


def load_and_crop_test_julia_patches(slide_data, patchsize=256, use_reduce=True):
    factor = 2
    if use_reduce is True:
        slide_data = resize(slide_data,
                            (slide_data.shape[0] // factor, slide_data.shape[1] // factor, slide_data.shape[2]),
                            preserve_range=True)
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
            augmented_img, augmented_manseg = fibseg_data_augmentation.augment1(image, manmask)
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
