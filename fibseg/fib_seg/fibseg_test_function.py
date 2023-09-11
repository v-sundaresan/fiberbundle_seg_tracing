from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import os
import glymur
from PIL import Image
from fibseg.fib_seg import (fibseg_model, fibseg_evaluate, fibseg_data_postprocessing)
from fibseg.utils import fibseg_utils
from skimage import morphology
from scipy import ndimage
from skimage.measure import regionprops, label

# =========================================================================================
# Truenet main test function
# Vaanathi Sundaresan
# 09-03-2021, Oxford
# =========================================================================================


def main(sect_name_dicts, eval_params, intermediate=False, model_dir=None,
         output_dir=None, verbose=False):
    """
    The main function for testing Truenet
    :param sect_name_dicts: list of dictionaries containing subject filepaths
    :param eval_params: dictionary of evaluation parameters
    :param intermediate: bool, whether to save intermediate results
    :param model_dir: str, filepath containing the test model
    :param output_dir: str, filepath for saving the output predictions
    :param verbose: bool, display debug messages
    """
    assert len(sect_name_dicts) > 0, "There must be at least 1 subject for testing."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if eval_params['Use_CPU']:
        device = torch.device("cpu")

    model_name = eval_params['Modelname']
    model_type = eval_params['Model_type']
    numchannels = eval_params['Num_channels']
    psize = eval_params['Patch_size']

    try:
        model_path = os.path.join(model_dir, model_name)
        try:
            state_dict = torch.load(model_path)
        except:
            state_dict = torch.load(model_path, map_location='cpu')

        for key, value in state_dict.items():
            if 'outconv' in key and 'weight' in key:
                nclass = state_dict[key].size()[0]
            if 'inpconv' in key and 'weight' in key:
                numchannels = value.size()[1]
        if model_type == 'unet':
            model = fibseg_model.EncDecClass(n_channels=numchannels, n_classes=nclass, init_channels=64,
                                         feat_channels=128, plane='axial')
        model.to(device=device)
        model = nn.DataParallel(model)
        model = fibseg_utils.loading_model(model_path, model)
    except ImportError:
        raise ImportError('In directory ' + model_dir + ', ' + model_name + '.pth' +
                          'does not appear to be a valid model file')

    if sect_name_dicts[0]['sect_path'] is None:
        raise ImportError('Section paths must be provided as the input')

    if verbose:
        print('Found' + str(len(sect_name_dicts)) + 'subjects', flush=True)

    eval_params['Num_channels'] = numchannels
    for sub in range(len(sect_name_dicts)):
        if verbose:
            print('Predicting output for subject ' + str(sub + 1) + '...', flush=True)

        test_sect_dict = [sect_name_dicts[sub]]
        basename = test_sect_dict[0]['basename']

        probs_combined = []
        sect_path = test_sect_dict[0]['sect_path']
        jp2 = glymur.Jp2k(sect_path)
        test_slice = jp2.read(rlevel=4)
        if verbose:
            print('Section shape: ')
            print(test_slice.shape)
        if intermediate:
            save_path = os.path.join(output_dir, 'Fibseg_input_testslice_' + basename + '.jpg')
            if verbose:
                print('Saving the intermediate raw probability map before preprocessing ...', flush=True)
            im = Image.fromarray(test_slice)
            im.save(save_path)

        brainmask = fibseg_utils.getting_brainmask_from_ascii(sect_name_dicts, output_dir)
        segoutpatches, inpimgpatches = fibseg_evaluate.evaluate_fibseg(test_slice, model, eval_params, device,
                                                                       modtype='unet', p_size=256, verbose=verbose)
        probs_out = fibseg_data_postprocessing.putting_test_patches_back_into_slides(test_sect_dict, segoutpatches,
                                                                                     patchsize=psize, use_reduce=False)
        probs_out = probs_out[1, :, :]
        prediction_bin = (probs_out > 0.3).astype(float)
        if intermediate:
            save_path = os.path.join(output_dir, 'Fibseg_predicted_initprobmap_' + basename + '.jpg')
            save_path_bin = os.path.join(output_dir, 'Fibseg_predicted_initbin_' + basename + '.jpg')
            if verbose:
                print('Saving the intermediate raw probability map before preprocessing ...', flush=True)
            im = Image.fromarray(probs_out)
            im.save(save_path)
            im_bin = Image.fromarray(prediction_bin)
            im_bin.save(save_path_bin)
        print('Prediction binary shapes')
        print(prediction_bin.shape)
        prediction_bin_bmask, diff_brainmask = fibseg_utils.applying_brainmask(prediction_bin, brainmask)
        final_predlabel, areas = fibseg_data_postprocessing.final_candidate_postprocessing(prediction_bin_bmask,
                                                                                           pred_thr=0.3,
                                                                                           area_thr=1000,
                                                                                           modeltype='unet')

        pred = (final_predlabel > 0).astype(float)
        imageg = 0.2989 * test_slice[:, :, 0] + 0.5870 * test_slice[:, :, 0] + 0.1140 * test_slice[:, :, 0]
        brain = (imageg ** 0.5 > 1).astype(float)
        str_el = morphology.disk(11)
        brainmap = ndimage.binary_opening(brain, structure=str_el).astype(np.float)
        vents = (brainmap == 0).astype(float) * (brainmask == 1).astype(float)
        if intermediate:
            save_path = os.path.join(output_dir, 'Fibseg_ventricles_slice_' + basename + '.jpg')
            if verbose:
                print('Saving the intermediate raw probability map before preprocessing ...', flush=True)
            im = Image.fromarray(vents)
            im.save(save_path)
        labvent = label(vents > 0)
        props = regionprops(labvent)
        areas = np.array([prop['area'] for prop in props])
        # print(areas)
        ventinds = np.where(areas > 50000)[0]
        # print(ventinds)
        vents = np.ones_like(vents)
        for idx in list(ventinds):
            vents[labvent == idx + 1] = 0
        bmask = 1 - (diff_brainmask > 0).astype(float) * vents
        # print(bmask.shape)
        bmask_dist_map = ndimage.morphology.distance_transform_edt(bmask)
        # row_upsample_factor = slide.shape[0] / target_map.shape[0]
        # col_upsample_factor = slide.shape[1] / target_map.shape[1]

        pred_label, pred_num = label(pred > 0, return_num=True)
        prprops = regionprops(pred_label)
        # print(len(prprops))
        prareas = [prprop['area'] for prprop in prprops]
        prbboxes = [prprop['bbox'] for prprop in prprops]
        prcents = [prprop['centroid'] for prprop in prprops]
        bmask_distances = []
        for pr in range(len(prprops)):
            cent = np.round(prcents[pr]).astype(int)
            dist_from_bmask = bmask_dist_map[cent[0], cent[1]]
            bmask_distances.append(dist_from_bmask)
        bmask_distances = np.array(bmask_distances)
        if verbose:
            print('SLIDE NAME: ' + test_sect_dict[0]['sect_path'])
            print('All brainmask distances: ')
            print(bmask_distances)
            print('Areas: ')
            print(prareas)

        pred_segmaps = (pred_label > 0).astype(float)
        pred_segmaps = (pred_segmaps > 0).astype(float)

        predmap = fibseg_data_postprocessing.final_filtering_stage(pred_segmaps, bmask_distances)

        save_path = os.path.join(output_dir, 'Fibseg_predicted_finaloutput_' + basename + '.jpg')
        if verbose:
            print('Saving the final prediction ...', flush=True)
        im = Image.fromarray(predmap)
        im.save(save_path)

    if verbose:
        print('Testing complete for all subjects!', flush=True)

