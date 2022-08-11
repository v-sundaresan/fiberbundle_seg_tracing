from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.utils.data import Dataset
import numpy as np
import random
from scipy.ndimage.interpolation import shift
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import rotate, zoom

# =========================================================================================
# Enc-dec dataset utility functions
# Vaanathi Sundaresan
# 09-03-2021, Oxford
# =========================================================================================


class HistTestDataset(Dataset):
    """This is a generic class for 2D segmentation datasets.
    :param data: stack of 3D slices N x C x H x W
    :param transform: transformations to apply.
    """
    def __init__(self, source_data, transform=None):
        self.source_data = torch.from_numpy(source_data).float()
        self.transform = transform

    def __getitem__(self, index):
        sourcex = self.source_data[index]

        data_dict = {
            'hist': sourcex
        }

        if self.transform:
            data_dict = self.transform(data_dict)

        return data_dict

    def __len__(self):
        return len(self.source_data)


class HistPriorDataset(Dataset):
    """This is a generic class for 2D segmentation datasets.
    :param data: stack of 3D slices N x C x H x W
    :param target: stack of 3D slices N x C x H x W
    :param transform: transformations to apply.
    """
    def __init__(self, source_data, target_data, prior_data, transform=None):
        self.source_data = torch.from_numpy(source_data).float()
        self.target_data = torch.from_numpy(target_data).float()
        self.prior_data = torch.from_numpy(prior_data).float()
        self.transform = transform  #?| This is where you can add augmentations

    def __getitem__(self, index):
        sourcex = self.source_data[index]
        targetx = self.target_data[index]
        priorx = self.prior_data[index]

        data_dict = {
            'hist': sourcex,
            'label': targetx,
            'prior': priorx
        }

        if self.transform:
            data_dict = self.transform(data_dict)

        return data_dict

    def __len__(self):
        return len(self.source_data)


class HistDataset(Dataset):
    """This is a generic class for 2D segmentation datasets.
    :param data: stack of 3D slices N x C x H x W
    :param target: stack of 3D slices N x C x H x W
    :param transform: transformations to apply.
    """
    def __init__(self, source_data, target_data, transform=None):
        self.source_data = torch.from_numpy(source_data).float()
        self.target_data = torch.from_numpy(target_data).float()
        self.transform = transform  # add augmentations

    def __getitem__(self, index):
        sourcex = self.source_data[index]
        targetx = self.target_data[index]

        data_dict = {
            'hist': sourcex,
            'label': targetx
        }

        if self.transform:
            data_dict = self.transform(data_dict)

        return data_dict

    def __len__(self):
        return len(self.source_data)


class HistTestDataset3D(Dataset):
    """This is a generic class for 2D segmentation datasets.
    :param data: stack of 3D slices N x C x H x W x D
    :param transform: transformations to apply.
    """
    def __init__(self, source_data, transform=None):
        self.source_data = torch.from_numpy(source_data).float()
        self.transform = transform

    def __getitem__(self, index):
        sourcex = self.source_data[index]

        data_dict = {
            'hist': sourcex
        }

        if self.transform:
            data_dict = self.transform(data_dict)

        return data_dict

    def __len__(self):
        return len(self.source_data)


class HistDataset3D(Dataset):
    """This is a generic class for 2D segmentation datasets.
    :param data: stack of 3D slices N x C x H x W x D
    :param target: stack of 3D slices N x C x H x W x D
    :param transform: transformations to apply.
    """
    def __init__(self, source_data, target_data, transform=None):
        self.source_data = torch.from_numpy(source_data).float()
        self.target_data = torch.from_numpy(target_data).float()
        self.transform = transform  #?| This is where you can add augmentations

    def __getitem__(self, index):
        sourcex = self.source_data[index]
        targetx = self.target_data[index]

        data_dict = {
            'hist': sourcex,
            'label': targetx
        }

        if self.transform:
            data_dict = self.transform(data_dict)

        return data_dict

    def __len__(self):
        return len(self.source_data)


class HistDatasetAug(Dataset):
    """This is a generic class for 2D segmentation datasets.
    :param data: stack of 3D slices N x C x H x W
    :param target: stack of 3D slices N x C x H x W
    :param transform: transformations to apply.
    """
    def __init__(self, source_data, target_data, transform=None):
        self.source_data = source_data
        self.target_data = target_data
        self.transform = transform  #?| This is where you can add augmentations

    def __getitem__(self, index):
        sourcex = self.source_data[index]
        targetx = self.target_data[index]

        data_dict = {
            'hist': sourcex,
            'label': targetx
        }

        if self.transform:
            transform_key = random.choice(list(self.transform))  # We randomly choose a transform from the dict.
            data_dict = self.transform[transform_key](data_dict)  # Apply the transform

        return data_dict

    def __len__(self):
        return len(self.source_data)


class HistDatasetAugOmni(Dataset):
    """This is a generic class for 2D segmentation datasets.
    :param data: stack of 3D slices N x C x H x W
    :param target: stack of 3D slices N x C x H x W
    :param transform: transformations to apply.
    """
    def __init__(self, source_data, transform=None):
        self.source_data = source_data
        self.transform = transform  #?| This is where you can add augmentations

    def __getitem__(self, index):
        sourcex = self.source_data[index]

        data_dict = {
            'hist': sourcex,
        }

        if self.transform:
            transform_key = random.choice(list(self.transform))  # We randomly choose a transform from the dict.
            data_dict = self.transform[transform_key](data_dict)  # Apply the transform

        return data_dict

    def __len__(self):
        return len(self.source_data)


class ToTensor(object):
    """Convert a PIL image or numpy array to a PyTorch tensor."""

    def __init__(self, labeled=True):
        self.labeled = labeled

    def __call__(self, sample):
        rdict = dict()
        rdict['hist'] = torch.from_numpy(sample['hist']).float()
        if self.labeled is True:
            rdict['label'] = torch.from_numpy(sample['label']).float()
        rdict['aug'] = torch.from_numpy(sample['aug']).float()

        return rdict


class RandomTranslation(object):
    """Translate the volume's values.
    :param degrees: maximum of translation values.
    """

    def __init__(self, offsets, segment=True, labeled=True):
        self.offsets = offsets
        self.labeled = labeled
        self.segment = segment
        self.order = 0 if self.segment is True else 5

    @staticmethod
    def get_params(offsets):  # Get random x and y offset values for translation
        offsetx = np.random.uniform(offsets[0], offsets[1])
        offsety = np.random.uniform(offsets[2], offsets[3])
        return offsetx, offsety

    def __call__(self, sample):
        rdict = dict()
        input_data = sample['hist']
        if len(sample['hist'].shape) != 3:  # C x X_dim x Y_dim
            raise ValueError("Input of RandomRotation3D should be a 3 dimensionnal tensor.")

        offsetx, offsety = self.get_params(self.offsets)
        translated_im = shift(input_data, (0, offsetx, offsety), order=self.order, mode='nearest')
        rdict['hist'] = sample['hist']
        if self.labeled is True:
            rdict['label'] = sample['label']
        rdict['aug'] = translated_im
        # Update the dictionary with transformed image and labels
        return rdict


class RandomRotation(object):
    """Make a rotation of the volume's values.
    :param degrees: Maximum rotation's degrees.
    """
    def __init__(self, degrees, segment=True, labeled=True):
        self.degrees = degrees
        self.labeled = labeled
        self.segment = segment
        self.order = 0 if self.segment is True else 5

    @staticmethod
    def get_params(degrees):  # Get random theta value for rotation
        angle = np.random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, sample):
        rdict = dict()
        input_data = sample['hist']
        if len(sample['hist'].shape) != 3:  # C x X_dim x Y_dim
            raise ValueError("Input of RandomRotation should be a 3 dimensionnal tensor.")

        angle = self.get_params(self.degrees)
        input_rotated = np.zeros(input_data.shape, dtype=input_data.dtype)

        # Rotation angle chosen at random and rotation happens only on XY plane for both image and label.
        for sh in range(input_data.shape[0]):
            input_rotated[sh, :, :] = rotate(input_data[sh, :, :], float(angle), reshape=False, order=self.order, mode='nearest')

        # Update the dictionary with transformed image and labels
        rdict['hist'] = sample['hist']
        rdict['aug'] = input_rotated
        if self.labeled is True:
            rdict['label'] = sample['label']
        return rdict


class ToTensor1(object):
    """Convert a PIL image or numpy array to a PyTorch tensor."""

    def __init__(self, labeled=True):
        self.labeled = labeled

    def __call__(self, sample):
        rdict = {}
        input_data = sample['hist']
        ret_input = torch.from_numpy(input_data).float()
        rdict['hist'] = ret_input
        if self.labeled:
            gt_data = sample['label']
            if gt_data is not None:
                ret_gt = torch.from_numpy(gt_data).float()
                rdict['label'] = ret_gt
        sample.update(rdict)
        return sample


class RandomRotation3D(object):
    """Make a rotation of the volume's values.
    :param degrees: Maximum rotation's degrees.
    """

    def __init__(self, degrees, labeled=True, segment=True):
        self.degrees = degrees
        self.labeled = labeled
        self.segment = segment
        self.order = 0 if self.segment is True else 5

    @staticmethod
    def get_params(degrees):  # Get random theta value for rotation
        angle = np.random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, sample):
        rdict = {}
        input_data = sample['hist']
        if len(sample['hist'].shape) != 4:  # C x X_dim x Y_dim x Z_dim
            raise ValueError("Input of RandomRotation3D should be a 4 dimensionnal tensor.")

        angle = self.get_params(self.degrees)
        input_rotated = np.zeros(input_data.shape, dtype=input_data.dtype)
        gt_data = sample['label'] if self.labeled else None
        gt_rotated = np.zeros(gt_data.shape, dtype=gt_data.dtype) if self.labeled else None

        # Rotation angle chosen at random and rotation happens only on XY plane for both image and label.
        for ch in range(input_data.shape[0]):
            for sh in range(input_data.shape[3]):
                input_rotated[ch, :, :, sh] = rotate(input_data[ch, :, :, sh], float(angle), reshape=False,
                                                     order=self.order, mode='nearest')
        if self.labeled:
            for sh in range(input_data.shape[3]):
                gt_rotated[:, :, sh] = rotate(gt_data[:, :, sh], float(angle), reshape=False, order=self.order,
                                              mode='nearest')
                gt_rotated = (gt_rotated > 0.4).astype(float)
            rdict['label'] = gt_rotated

        # Update the dictionary with transformed image and labels
        rdict['hist'] = input_rotated
        sample.update(rdict)
        return sample


class RandomTranslation3D(object):
    """Translate the volume's values.
    :param degrees: maximum of translation values.
    """

    def __init__(self, offsets, labeled=True, segment=True):
        self.offsets = offsets
        self.labeled = labeled
        self.segment = segment
        self.order = 0 if self.segment == True else 5

    @staticmethod
    def get_params(offsets):  # Get random x and y offset values for translation
        offsetx = np.random.uniform(offsets[0], offsets[1])
        offsety = np.random.uniform(offsets[2], offsets[3])
        return offsetx, offsety

    def __call__(self, sample):
        rdict = {}
        input_data = sample['hist']
        if len(sample['hist'].shape) != 4:  # C x X_dim x Y_dim x Z_dim
            raise ValueError("Input of RandomRotation3D should be a 4 dimensionnal tensor.")

        offsetx, offsety = self.get_params(self.offsets)
        translated_im = np.zeros(input_data.shape)
        translated_im[0, :, :, :] = shift(input_data[0, :, :, :], (offsetx, offsety, 0), order=self.order,
                                          mode='nearest')
        translated_im[1, :, :, :] = shift(input_data[1, :, :, :], (offsetx, offsety, 0), order=self.order,
                                          mode='nearest')
        translated_im[2, :, :, :] = shift(input_data[2, :, :, :], (offsetx, offsety, 0), order=self.order,
                                          mode='nearest')
        rdict['hist'] = translated_im
        if self.labeled:
            gt_data = sample['label']
            translated_label = shift(gt_data, (offsetx, offsety, 0), order=self.order,
                                                 mode='nearest')
            rdict['label'] = translated_label
        sample.update(rdict)
        return sample









