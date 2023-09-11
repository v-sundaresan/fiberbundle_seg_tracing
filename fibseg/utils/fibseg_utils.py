from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from skimage.transform import resize
from skimage.measure import regionprops, label
from scipy import ndimage
from skimage import exposure, morphology, filters
import math
import os
import json
import glymur
from slider import chart_reg
from skimage import draw, color
from slider import util

#=========================================================================================
# Fiber Bundle segmentation tool - utils function
# Vaanathi Sundaresan
# 10-08-2023
#=========================================================================================


def getting_brainmask_from_ascii(sect_name_dicts, output_dir):
    sect_name = sect_name_dicts['sect_path']
    ascii_name = sect_name_dicts['ascii_path']
    resolution = 0.0004
    out = chart_reg.register_chart_to_slide(ascii_name, sect_name, resolution, output_dir)
    f = open(os.path.join(output_dir, 'contours.json'))
    data = json.load(f)
    jp2 = glymur.Jp2k(sect_name)
    img_orig = jp2.read(rlevel=4)
    img = color.rgb2gray(img_orig)
    height, width = img.shape
    required_names = ['Outline_', 'Outline']
    # ['Fiber dense area', 'Fiber moderate area', 'Fiber light area','Fiber bundle light', 'Fiber bundle medium',
    # 'Fiber bundle dense'], ['Fiber_dense_area_', 'Fiber_moderate_area_', 'Fiber_light_area_', 'Fiber_bundle_light_',
    # 'Fiber_bundle_medium_', 'Fiber_bundle_dense_']
    indices = [1, 1]
    factor = 16
    for reg in range(len(required_names)):
        mask = np.zeros([height // factor, width // factor])
        for dat in data:
            if required_names[reg] in dat['name']:
                coords = np.fliplr(np.array(dat['xy'])) / (0.0064 * factor)
                image = draw.polygon2mask((height // factor, width // factor), coords)
                mask = mask + (image * indices[reg])
        mask = resize(mask, (height, width), preserve_range=True)
    return mask


def getting_priormap_slide(priorvol, slid, bmask1, img_shape, config1):
    row_ds_factor = config1['row_ds_factor']
    specific_slide = priorvol[slid, :, :]
    cropped_bmask_r = resize(bmask1, (bmask1.shape[0]//row_ds_factor,
                                      bmask1.shape[1]//row_ds_factor), preserve_range=True)
    start_row = priorvol.shape[1] // 2 - cropped_bmask_r.shape[0] // 2
    end_row = priorvol.shape[1] // 2 - cropped_bmask_r.shape[0] // 2 + cropped_bmask_r.shape[0]
    start_col = priorvol.shape[2] // 2 - cropped_bmask_r.shape[1] // 2
    end_col = priorvol.shape[2] // 2 - cropped_bmask_r.shape[1] // 2 + cropped_bmask_r.shape[1]
    priormap_slide1 = specific_slide[start_row:end_row, start_col:end_col]
    prior_slide = resize(priormap_slide1, (img_shape[2], img_shape[3]), preserve_range=True)
    return prior_slide


def ignore_other_bundles(predimg, labeled_target, select_bundle='dense'):
    """
    :param predimg: predicted output
    :param labeled_target: labelled map of manual segmntation
    :param select_bundle: Bundle group to select for evaluation: 'light', 'moderate', 'dense'
    :return:
    """
    labeled_pred, lab_pred_num = label(predimg, return_num=True)
    print(lab_pred_num)
    if select_bundle == 'dense':
        select_idx = 3
        ignore_idx = [1, 2]
    elif select_bundle == 'moderate':
        select_idx = 2
        ignore_idx = [1, 3]
    else:
        select_idx = 1
        ignore_idx = [2, 3]

    regions_ignored1 = np.union1d(labeled_pred[labeled_target == ignore_idx[0]], [])
    print(regions_ignored1)
    regions_ignored1 = np.setdiff1d(np.array(regions_ignored1), 0)
    print(regions_ignored1)
    regions_ignored2 = np.union1d(labeled_pred[labeled_target == ignore_idx[1]], [])
    print(regions_ignored2)
    regions_ignored2 = np.setdiff1d(np.array(regions_ignored2), 0)
    print(regions_ignored2)
    regions_ignored = np.union1d(regions_ignored1, regions_ignored2)
    print(regions_ignored)

    regions_selected = np.union1d(labeled_pred[labeled_target == select_idx], [])
    print(regions_selected)
    dense_regions_selected = np.setdiff1d(np.array(regions_selected), 0)
    print(dense_regions_selected)

    all_regions_selected = np.setdiff1d(np.arange(1, lab_pred_num+1), regions_ignored)
    print(all_regions_selected)
    all_regions_selected = np.union1d(dense_regions_selected, all_regions_selected)
    print(all_regions_selected)

    new_labelled_pred = np.zeros_like(labeled_pred)
    for idxs in range(len(all_regions_selected)):
        new_labelled_pred[labeled_pred == all_regions_selected[idxs]] = 1

    print('Total regions in the predicted output: ' + str(lab_pred_num))
    print('Total number of selected regions in the predicted output: ' + str(len(all_regions_selected)))
    return (new_labelled_pred > 0).astype(float), (labeled_target == select_idx).astype(float)


def applying_brainmask(prediction, brain_mask):
    if len(prediction.shape) == 4:
        prediction = prediction[0, 1, :, :]
    elif len(prediction.shape) == 3:
        try:
            prediction = prediction[1, :, :]
        except:
            prediction = prediction[0, :, :]

    if len(brain_mask.shape) > 2:
        brain_mask = brain_mask[0, :, :]

    if brain_mask.shape[0] != brain_mask.shape[0]:
        brain_mask = resize(brain_mask, prediction.shape, preserve_range=True)

    brain_mask = (brain_mask > 0.5).astype(float)
    strel = morphology.disk(21)
    final_brainmask = ndimage.binary_erosion(brain_mask > 0, structure=strel).astype(np.float)
    constrained_prediction = prediction * final_brainmask
    return constrained_prediction, (brain_mask - final_brainmask).astype(float)


def dist(cent1, cent2):
    return math.sqrt((cent1[0] - cent2[0])**2 + (cent1[1] - cent2[1])**2)


def applying_priormap_constraints(predictionbin, priormap, priorcons_type='intersect',
                                  priordist_thr=100):
    """
    :param predictionbin: binarised prediction slide [H, W] or [1, H, W]
    :param priormap: priormap slide [H, W] or [1, H, W]
    :param priorcons_type: 'intersect' or 'centdistance'
    :param priordist_thr:
    :return:
    """
    priormap = (priormap > 0.4).astype(float)

    if len(priormap.shape) > 2:
        priormap = priormap[0, :, :]

    if len(predictionbin.shape) > 2:
        predictionbin = predictionbin[0, :, :]

    if priormap.shape[0] != predictionbin.shape[0]:
        priormap = resize(priormap, predictionbin.shape, preserve_range=True)

    if priorcons_type == 'intersect':
        constrained_prediction = predictionbin * priormap
    else:
        labelprior, numcomp = label(priormap, return_num=True)
        prps = regionprops(labelprior)
        prior_cents = []
        for prp in range(len(prps)):
            prior_cents.append(prps[prp]['centroid'])
        labelpred, numcomp = label(predictionbin, return_num=True)
        prps = regionprops(labelpred)
        pred_cents = []
        for prp in range(len(prps)):
            pred_cents.append(prps[prp]['centroid'])

        cand_dists = []
        for cnt in pred_cents:
            if not prior_cents:
                icand_dist = 0
            else:
                icand_dists = [dist(cnt, pcent) for pcent in prior_cents]
                icand_dist = min(icand_dists)
            cand_dists.append(icand_dist)

        cand_dists = np.array(cand_dists)
        pass_cand_ids = np.where(cand_dists < priordist_thr)[0]
        constrained_prediction = np.zeros_like(predictionbin)
        for idxs in list(pass_cand_ids):
            weighted_ican = (labelpred == idxs + 1).astype(float) * cand_dists[idxs]
            constrained_prediction = constrained_prediction + weighted_ican
    return constrained_prediction, cand_dists, priormap

