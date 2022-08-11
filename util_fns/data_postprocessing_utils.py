from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import nibabel as nib
import random
from skimage.transform import resize
import hist_seg_patches_augmentation

# =========================================================================================
# Data postprocessing code for histology data (nifti files)
# Vaanathi Sundaresan
# 3-31-2022, MGH Martinos MA, USA
# =========================================================================================


def putting_test_patches_back_into_slides(slidename, prob_patches, patchsize=256, use_reduce=True, factor=2):
    slide_data = np.load(slidename).astype('float32')
    if use_reduce is True:
        slide_data = resize(slide_data,
                            (slide_data.shape[0] // factor, slide_data.shape[1] // factor, slide_data.shape[2]),
                            preserve_range=True)
    slide_data = np.squeeze(slide_data)
    prob_slide = np.zeros([1, prob_patches.shape[1], slide_data.shape[0], slide_data.shape[1]])
    height, width, _ = slide_data.shape
    num_along_height = np.ceil(height / patchsize).astype(int)
    num_along_width = np.ceil(width / patchsize).astype(int)
    num_patches = np.ceil(num_along_height * num_along_width).astype(int)
    if num_patches != prob_patches.shape[0]:
        ValueError('The number of patches and the probability outputs do not match!')
    else:
        count = 0
        for row in range(num_along_height):
            for col in range(num_along_width):
                start_row = np.amax([row * patchsize, 0])
                end_row = np.amin([(row + 1) * patchsize, slide_data.shape[0]])
                start_col = np.amax([col * patchsize, 0])
                end_col = np.amin([(col + 1) * patchsize, slide_data.shape[1]])
                tmp_patch = slide_data[start_row:end_row, start_col:end_col, :]
                tmph, tmpw, _ = tmp_patch.shape
                print('###############PATCHSIZE DEBUGGING###################')
                print(tmp_patch.shape)
                print(prob_patches.shape)
                print(tmph, tmpw)
                temp_prob = prob_patches[count, :, :tmph, :tmpw]
                prob_slide[0, :, start_row:end_row, start_col:end_col] = temp_prob
                count += 1

    return prob_slide
