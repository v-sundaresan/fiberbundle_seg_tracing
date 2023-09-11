from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from scipy.ndimage.interpolation import shift
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import rotate
from skimage.util import random_noise

#=========================================================================================
# Fibseg augmentations function
# Vaanathi Sundaresan
# 11-08-2023
#=========================================================================================


##########################################################################################
# Define transformations with distance maps
##########################################################################################

def translate1(image, label):
    """
    :param image: mod1
    :param label: manual mask
    :return:
    """
    offsetx = random.randint(-25, 25)
    offsety = random.randint(-25, 25)
    is_seg = False
    order = 0 if is_seg is True else 5
    translated_im = shift(image, (offsetx, offsety, 0), order=order, mode='nearest')
    translated_label = shift(label, (offsetx, offsety, 0), order=order, mode='nearest')
    return translated_im, translated_label


def flip_horizontal1(image, label):
    """
        :param image: mod1
        :param label: manual mask
        :return:
        """
    return image[:, :, ::-1], label[:, :, ::-1]


def rotate1(image, label):
    """
    :param image: mod1
    :param label: manual mask
    :return:
    """
    theta = random.uniform(-15, 15)
    is_seg = False
    order = 0 if is_seg is True else 5
    new_img = rotate(image, float(theta), axes=(2, 1), reshape=False, order=order, mode='nearest')
    new_lab = rotate(label, float(theta), axes=(2, 1), reshape=False, order=order, mode='nearest')
    new_lab = (new_lab > 0.5).astype(float)
    return new_img, new_lab


def blur1(image, label):
    """
    :param image: mod1
    :param label: manual mask
    :return:
    """
    sigma = random.uniform(0.1, 0.2)
    new_img = gaussian_filter(image, sigma)
    return new_img, label


def add_noise1(image, label):
    """
    :param image: mod1
    :param label: manual mask
    :return:
    """
    new_img = random_noise(image, clip=False)
    return new_img, label


##########################################################################################
# Define transformations with 1 modality
##########################################################################################


def augment1(image, label):
    # Applies a random number of the possible transformations to the inputs.
    """
    :param image: mod1
    :param label: manual mask
    :return:
    """
    if len(image.shape) == 3:
        # Add to the available transformations any functions you want to be applied
        available_transformations = {'noise': add_noise1, 'translate': translate1,
                                     'rotate': rotate1, 'blur': blur1, 'flip_hor': flip_horizontal1}
        # Decide how many of these transformations to apply
        num_transformations_to_apply = random.randint(1, len(available_transformations))
        num_transformations = 0
        transformed_image = None
        transformed_label = None

        while num_transformations <= num_transformations_to_apply:
            # choose which transformations to apply at random
            key = random.choice(list(available_transformations))
            transformed_image, transformed_label = available_transformations[key](image, label)
            num_transformations += 1
        return transformed_image, transformed_label
    else:
        raise Exception('Invalid dimensions for image augmentation - currently only supported in 3d')


