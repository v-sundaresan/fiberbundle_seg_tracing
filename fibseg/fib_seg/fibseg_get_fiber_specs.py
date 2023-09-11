from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import rotate
from skimage.util import random_noise
from skimage.measure import regionprops, label
from scipy import ndimage
from skimage import exposure, morphology, filters

#=========================================================================================
# Fibseg - extracting fiber properties function
# Vaanathi Sundaresan
# 11-08-2023
#=========================================================================================


def get_fibre_properties(slide_crop, mask_crop):
    slide_crop_masked = slide_crop * mask_crop
    slide_crop_masked = slide_crop_masked / np.amax(slide_crop_masked)
    slide_crop_masked = slide_crop_masked ** 2
    slide_crop_masked = slide_crop_masked / np.amax(slide_crop_masked)
    slide_cm_eq = exposure.equalize_hist(slide_crop_masked)  # , clip_limit=0.03)
    thr_val = np.percentile(slide_cm_eq[mask_crop > 0], 95)
    final_fiber_map = slide_cm_eq > thr_val
#     str_el = morphology.square(3)
#     final_fiber_map = ndimage.binary_opening(final_fiber_map, structure=str_el).astype(np.float)
    finalfibmap_labelled = label(final_fiber_map > 0)
    pprops = regionprops(finalfibmap_labelled)
    print('Total number of fibers before filtering: ' + str(len(pprops)))
    ellipticity = [pprop.eccentricity for pprop in pprops]
    areas1 = [pprop.area for pprop in pprops]
    if len(areas1) == 0:
        print('The area array is empty!')
    else:
        print(min(areas1), max(areas1))

    new_fibre_map = np.zeros_like(final_fiber_map)
    if len(areas1) == 0:
        new_pprops = regionprops(label(new_fibre_map))
        maj_axis_lengths = []
        orientations = []
        total_area = 0
    else:
        for i in range(len(ellipticity)):
            if ellipticity[i] > 0.5 and areas1[i] > 10:
                new_fibre_map = new_fibre_map + (finalfibmap_labelled == i + 1).astype(float)

        new_pprops = regionprops(label(morphology.skeletonize(new_fibre_map) > 0))
        maj_axis_lengths = [pprop.area for pprop in new_pprops]
        orientations = np.array([pprop.orientation * 180 / 3.14 for pprop in new_pprops])
        orientations[orientations < 0] = orientations[orientations < 0] + 180
        total_area = np.sum((new_fibre_map > 0).astype(float))
    print('Total number of fibers after filtering: ' + str(len(new_pprops)))
    return (new_fibre_map > 0).astype(float), total_area, len(new_pprops), maj_axis_lengths, orientations


def get_fibre_specs(new_fibre_map, mask_crop):
    new_fibre_map = new_fibre_map * mask_crop
    new_pprops = regionprops(label(morphology.skeletonize(new_fibre_map) > 0))
    maj_axis_lengths = [pprop.area for pprop in new_pprops]
    orientations = np.array([pprop.orientation * 180 / 3.14 for pprop in new_pprops])
    orientations[orientations < 0] = orientations[orientations < 0] + 180
    total_area = np.sum((new_fibre_map > 0).astype(float))
    return total_area, len(new_pprops), maj_axis_lengths, orientations