import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import rotate
from skimage.util import random_noise
from skimage.transform import resize
from skimage.measure import regionprops, label
from scipy import ndimage
from skimage import exposure, morphology, filters

# =========================================================================================
# Fibseg postprocessing function
# Vaanathi Sundaresan
# 10-08-2023
# =========================================================================================


def final_candidate_postprocessing(prediction, pred_thr=0.3, area_thr=2000, modeltype='unet'):
    # perform area based filtering
    if len(prediction.shape) == 3:
        prediction = prediction[1, :, :]
    # Do morphological operations
    if modeltype == 'unet':
        strel = morphology.disk(15)
        cleaned_pred = ndimage.binary_dilation(prediction > pred_thr, structure=strel).astype(np.float)
        strel = morphology.disk(9)
        final_prediction_map = ndimage.binary_erosion(cleaned_pred > 0, structure=strel).astype(np.float)
        final_prediction_map = ndimage.binary_fill_holes(final_prediction_map > 0)
    else:
        strel = morphology.disk(7)
        cleaned_pred = ndimage.binary_dilation(prediction > pred_thr, structure=strel).astype(np.float)
        strel = morphology.disk(2)
        final_prediction_map = ndimage.binary_erosion(cleaned_pred > 0, structure=strel).astype(np.float)
        final_prediction_map = ndimage.binary_fill_holes(final_prediction_map > 0)
    labelpred, nlab = label(final_prediction_map > 0, return_num=True)
    prps = regionprops(labelpred)
    areas1 = [prop['area'] for prop in prps]
    ids = np.where(np.array(areas1) > area_thr)[0]
    cleaned_pred = np.zeros_like(prediction)
    for idxs in list(ids):
        cleaned_pred = cleaned_pred + (labelpred == idxs + 1).astype(float) * (idxs + 1)

    return cleaned_pred, areas1


def final_filtering_stage(predimage, bdists):
    predlabel, numpredlabs = label(predimage, return_num=True)
    if numpredlabs != len(bdists):
        AssertionError('Number of regions in the map and distance array (from bmask) do match!')

    dist_indices = np.where(bdists > 200)[0]
    prps = regionprops(predlabel)
    areas1 = np.array([prop.area for prop in prps])
    if len(areas1) > 0:
        selected_areas = areas1[dist_indices]
        area_dist_indices = np.where(selected_areas > 5000)[0]
        big_area_indices = np.where(areas1 > 1.1e5)[0]
        sel_indices = list(np.union1d(area_dist_indices, big_area_indices))
        new_pred = np.zeros_like(predimage)
        for j in sel_indices:
            new_pred[predlabel == j + 1] = 1
    else:
        new_pred = (predimage > 0).astype(float)

    return new_pred


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
