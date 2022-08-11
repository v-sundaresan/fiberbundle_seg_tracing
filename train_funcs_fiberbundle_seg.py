# Utilities for training and validating the semi-sup temporal ensembling model
# Vaanathi Sundaresan, 2022
# arXiv:2208.03569
########################################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
import os
import util_fns.data_preparation_utils as data_preparation_utils
import util_fns.data_postprocessing_utils as data_postprocessing_utils
from test_funcs_fiberbundle_sg import test_histseg_temp_ensembling
from util_fns.utils import *
from util_fns.dataset_utils import *
from model.model_utils import *
from model.enc_dec_model_wo_lastconv import EncDec, EncDecClass
from torch.utils.data import DataLoader, Dataset
import glob
import Loss.loss_functions as loss_functions
########################################################################################################


def multiclass_dice_coeff(inp, tar):
    """
    Calculating Dice similarity coefficient
    :param inp: Input tensor
    :param tar: Target tensor
    :return: Dice value (scalar)
    """
    smooth = 1.
    pred_vect = inp.contiguous().view(-1)
    target_vect = tar.contiguous().view(-1)
    intersection = (pred_vect * target_vect).sum()
    dice = (2. * intersection + smooth) / (torch.sum(pred_vect) + torch.sum(target_vect) + smooth)
    return dice


def minmaxnorm(data):
    """
    :param data: Input tensor
    :return:
    """
    data = data - np.min(data)
    data = data/np.amax(data)
    return data


def get_label_from_prevepochs(oldmodels, trainnames, device, psize=256):
    """
    :param oldmodels: List of models
    :param trainnames: List of training names (strings)
    :param device
    :param psize
    :return:
    """
    batch_size = 4
    ldata = []
    for omodel in oldmodels:
        lab_data_patches, _, _ = test_histseg_temp_ensembling(trainnames, omodel, batch_size, device, type='unet',
                                                       p_size=psize)
        lab_data = data_postprocessing_utils.putting_test_patches_back_into_slides(trainnames, lab_data_patches,
                                                                                   patchsize=psize)
        lab_data[:, 1, :, :] = lab_data[:, 1, :, :] / np.amax(lab_data[:, 1, :, :])
        ldata.append(lab_data)
    ldata = np.array(ldata)
    print('INSIDE PREV EPOCH LOOP FOR OMNI_TRAINING')
    print(ldata.shape)
    ldata = np.mean(ldata, axis=0)
    return ldata


def train_histseg_temp_ensembling(train_hist_names, val_hist_names, save_checkpoint=True, fold_id=0, save_weights=True,
                                  save_case='best', verbose=True, dir_checkpoint=None):
    """
    :param train_hist_names: list of training data file paths
    :param val_hist_names: list of validation data file paths
    :param save_checkpoint: boolean
    :param fold_id: index for fold validation 0...(N-1) for N-fold validation
    :param save_weights: boolean
    :param save_case: boolean
    :param verbose: boolean
    :param dir_checkpoint: output directory for saving model (string)
    :return:
    """
    batch_size = 4
    patch_size = 256
    num_epochs = 285
    batch_factor = 1
    patience = 25
    save_resume = True
    lrt = 0.001
    nclass = 2
    initial_chs = 64

    composed_transform1 = transforms.Compose([
        RandomTranslation([-10, 10, -10, 10], labeled=True),
        ToTensor(labeled=True),
    ])
    composed_transform2 = transforms.Compose([
        RandomRotation([-30, 30], labeled=True),
        ToTensor(labeled=True),
    ])
    transformations = {'transform1': composed_transform1, 'transform2': composed_transform2}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncDecClass(n_channels=3, n_classes=nclass, init_channels=initial_chs, feat_channels=128,
                        plane='axial')
    model.to(device=device, device_ids=[0])
    model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=lrt, eps=1e-04)
    # recon_criterion = nn.MSELoss()
    wmconcriterion = loss_functions.SupConLoss(temperature=0.5)
    seg_criterion = loss_functions.FocalLoss(gamma=2, alpha=0.25)
    criterion = [seg_criterion, wmconcriterion]

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20], gamma=0.1, last_epoch=-1)

    early_stopping = EarlyStoppingModelCheckpointing(patience, verbose=verbose)

    num_iters = max(len(train_hist_names) // batch_factor, 1)

    losses_train = []
    losses_val = []
    dice_val = []
    best_val_dice = 0
    dir_checkpoint1 = '/autofs/space/nyx_002/users/vsundaresan/models/initial_seg/'
    start_epoch = 1
    if save_resume:
        try:
            if dir_checkpoint is not None:
                ckpt_path = os.path.join(dir_checkpoint1, 'tmp_model_unet_focal_sscon_TE' + str(fold_id) + '.pth')
            else:
                ckpt_path = os.path.join(os.getcwd(), 'tmp_model_unet_focal_sscon_TE' + str(fold_id) + '.pth')
            checkpoint_resumetraining = torch.load(ckpt_path)
            model.load_state_dict(checkpoint_resumetraining['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint_resumetraining['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_resumetraining['scheduler_state_dict'])
            start_epoch = checkpoint_resumetraining['epoch'] + 1
            losses_train = checkpoint_resumetraining['loss_train']
            losses_val = checkpoint_resumetraining['loss_val']
            dice_val = checkpoint_resumetraining['dice_val']
            best_val_dice = checkpoint_resumetraining['best_val_dice']
            print(start_epoch, flush=True)
        except:
            if dir_checkpoint is not None:
                ckpt_path = os.path.join(dir_checkpoint1, 'tmp_model_unet_focal_sscon_' + str(fold_id) + '.pth')
            else:
                ckpt_path = os.path.join(os.getcwd(), 'tmp_model_unet_focal_sscon_' + str(fold_id) + '.pth')
            checkpoint_resumetraining = torch.load(ckpt_path)
            model.load_state_dict(checkpoint_resumetraining['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint_resumetraining['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_resumetraining['scheduler_state_dict'])
            start_epoch = checkpoint_resumetraining['epoch'] + 1
            losses_train = checkpoint_resumetraining['loss_train']
            losses_val = checkpoint_resumetraining['loss_val']
            dice_val = checkpoint_resumetraining['dice_val']
            best_val_dice = checkpoint_resumetraining['best_val_dice']
            print(start_epoch, flush=True)

    old_model = model  # Model at N-1 epoch
    old_model1 = model  # Model at N-2 epoch
    print('Training started!!.......................................')
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        running_seg_loss = 0.0
        running_wm_loss = 0.0
        batch_count = 0
        old_model2 = old_model1  # Model at N-3 epoch
        old_model1 = old_model  # Model at N-2 epoch
        old_model = model  # Model at N-1 epoch
        old_model.eval()
        old_model1.eval()
        old_model2.eval()
        print('Epoch: ' + str(epoch) + ' starting!..............................')
        for i in range(num_iters):
            hist_trainnames = train_hist_names[i * batch_factor:(i + 1) * batch_factor]

            print('Training and validation hist/label files listing...................................')
            print(hist_trainnames)

            lab_data = get_label_from_prevepochs([old_model, old_model1, old_model2], hist_trainnames[0], device,
                                                 psize=1024)
            lab_data = np.argmax(lab_data, axis=1)  # N x H x W
            np.save(dir_checkpoint1 + 'labels_img_TE.npy', lab_data)
            [hist_train, lab_train] = data_preparation_utils.load_and_crop_TEtraining_regions(hist_trainnames,
                                                                                              lab_data[0, :, :],
                                                                                              num_patches=8,
                                                                                              patchsize=patch_size,
                                                                                              augment=False)
            np.save(dir_checkpoint1 + 'data_TE.npy', hist_train)
            np.save(dir_checkpoint1 + 'labels_TE.npy', lab_train)
            hist_train = hist_train - np.amin(hist_train)
            hist_train = hist_train / np.amax(hist_train)
            train_dataset_dict = HistDatasetAug(hist_train, lab_train, transformations)

            train_dataloader = DataLoader(train_dataset_dict, batch_size=batch_size, shuffle=True, num_workers=0)

            del hist_train
            del lab_train
            del lab_data

            for batch_idx, train_dict in enumerate(train_dataloader):
                histx = train_dict['hist']
                laby = train_dict['label']
                histx_aug = train_dict['aug']

                histx = torch.cat([histx, histx_aug], dim=0)

                print('Training dimensions.......................................')
                print(histx.shape)
                print(laby.shape)
                histx = histx.to(device=device, dtype=torch.float32)
                laby = laby.to(device=device, dtype=torch.float32)

                model.train()
                optimizer.zero_grad()
                predy, features = model.forward(histx, initial_chs)

                print('predicted target dimensions........')
                print(predy.size())
                bsz = laby.shape[0]
                seg_loss = seg_criterion(predy[:bsz, :, :, :], laby)
                predy = F.softmax(predy, dim=1)
                predy = predy.contiguous().view(-1, 2)
                predy = torch.argmax(predy, dim=1).double()

                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                wm_loss = wmconcriterion(features)

                running_seg_loss += seg_loss.item()
                running_wm_loss += wm_loss.item()
                total_loss = seg_loss + wm_loss
                total_loss.backward()
                optimizer.step()

                del laby
                del predy
                del features
                del f1
                del f2

                if batch_idx % 10 == 0:
                    print('Train Mini-batch: {} out of Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        (i + 1), epoch, (batch_idx + 1) * len(histx), len(train_dataloader.dataset),
                                        100. * (batch_idx + 1) / len(train_dataloader), total_loss.item()),
                        flush=True)
                del histx
                batch_count += 1
        lab_data = get_label_from_prevepochs([old_model, old_model1, old_model2], val_hist_names, device,
                                             psize=1024)
        lab_data = np.argmax(lab_data, axis=1)  # N x H x W

        [hist_val, lab_val] = data_preparation_utils.load_and_crop_TEtraining_regions(val_hist_names,
                                                                                      lab_data[0, :, :],
                                                                                      num_patches=8,
                                                                                      patchsize=patch_size,
                                                                                      augment=False)
        hist_val = hist_val - np.amin(hist_val)
        hist_val = hist_val / np.amax(hist_val)
        val_dataset_dict = HistDatasetAug(hist_val, lab_val, transformations)
        val_dataloader = DataLoader(val_dataset_dict, batch_size=4, shuffle=False, num_workers=0)

        del hist_val
        del lab_val
        del lab_data
        # del lab_data_patches
        val_av_loss, val_av_dice = validate_histseg_temp_ensembling(val_dataloader, model, old_model, device,
                                                                    criterion, nclass, initial_chs)
        del val_dataset_dict
        del val_dataloader
        scheduler.step()

        avseg_loss = (running_seg_loss / batch_count)  # .detach().cpu().numpy()
        avwm_loss = (running_wm_loss / batch_count)  # .detach().cpu().numpy()
        print('Training set: Average loss: ', avseg_loss, avwm_loss, flush=True)
        losses_train.append([avseg_loss, avwm_loss])
        losses_val.append(val_av_loss)
        dice_val.append(val_av_dice)

        if epoch % 20 == 0:
            if save_checkpoint:
                try:
                    os.mkdir(dir_checkpoint)
                except OSError:
                    pass
                torch.save(model.state_dict(), dir_checkpoint + f'CP_epoch{epoch+1}_focal_sscon_TE_' + str(fold_id) + '.pth')

        if save_resume:
            if dir_checkpoint is not None:
                ckpt_path = os.path.join(dir_checkpoint, 'tmp_model_unet_focal_sscon_TE_' + str(fold_id) + '.pth')
            else:
                ckpt_path = os.path.join(os.getcwd(), 'tmp_model_unet_focal_sscon_TE_' + str(fold_id) + '.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss_train': losses_train,
                'loss_val': losses_val,
                'dice_val': dice_val,
                'best_val_dice': best_val_dice
            }, ckpt_path)

        if save_checkpoint:
            np.savez(os.path.join(dir_checkpoint, 'losses_unet_focal_sscon_TE_' + str(fold_id) + '.npz'), train_loss=losses_train,
                     val_loss=losses_val)
            np.savez(os.path.join(dir_checkpoint, 'validation_unet_focal_sscon_TE_' + str(fold_id) + '.npz'), dice_mse_val=dice_val)

        # early_stopping(val_av_loss, val_av_dice, best_val_dice, model, epoch, optimizer, scheduler, av_loss,
        #               weights=save_weights, checkpoint=save_checkpoint, save_condition=save_case,
        #               model_path=dir_checkpoint)

        if val_av_dice > best_val_dice:
            best_val_dice = val_av_dice

        # if early_stopping.early_stop:
        #    print('Patience Reached - Early Stopping Activated', flush=True)
        #    if save_resume:
        #        if dir_checkpoint is not None:
        #            ckpt_path = os.path.join(dir_checkpoint, 'tmp_model.pth')
        #        else:
        #            ckpt_path = os.path.join(os.getcwd(), 'tmp_model.pth')
        #        os.remove(ckpt_path)
        #    return model
        #             sys.exit('Patience Reached - Early Stopping Activated')

        torch.cuda.empty_cache()  # Clear memory cache

    # if save_resume:
    # if dir_checkpoint is not None:
    #    ckpt_path = os.path.join(dir_checkpoint, 'tmp_model.pth')
    # else:
    #    ckpt_path = os.path.join(os.getcwd(), 'tmp_model.pth')
    # os.remove(ckpt_path)

    return model


def validate_histseg_temp_ensembling(val_dataloader, model, old_model, device, criterion, nclass, initial_chnls):
    model.eval()
    old_model.eval()
    [seg_criterion, wm_criterion] = criterion
    count = 0
    loss_val = 0
    mse_val = 0
    for batch_val_idx, val_dict in enumerate(val_dataloader):
        histvx = val_dict['hist']
        # labvy = val_dict['label']
        labvy, _ = old_model(histvx, initial_chnls)
        softmax = nn.Softmax()
        labvy = softmax(labvy)
        labvy = labvy.cpu().detach().numpy()
        labvy = np.argmax(labvy, axis=1).astype('float')
        labvy = torch.from_numpy(labvy).float()
        histvx_aug = val_dict['aug']

        histvx = torch.cat([histvx, histvx_aug], dim=0)
        bsz = labvy.shape[0]
        histvx = histvx.to(device=device, dtype=torch.float32)
        labvy = labvy.to(device=device, dtype=torch.float32)
        predvy, features = model(histvx, initial_chnls)
        predvy = predvy[:bsz, :, :, :]
        val_seg_loss = seg_criterion(predvy, labvy)
        predvy = F.softmax(predvy, dim=1)
        predvy = predvy.contiguous().view(-1, nclass)
        predvy = torch.argmax(predvy, dim=1).double()
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        val_wm_loss = wm_criterion(features)

        total_val_loss = val_seg_loss + val_wm_loss
        loss_val += total_val_loss.item()

        labvy_vector = labvy.contiguous().view(-1)
        dice_coeff = multiclass_dice_coeff(predvy, labvy_vector)

        count += 1

    av_val_loss = loss_val / count
    av_dice_value = dice_coeff / count
    return av_val_loss, av_dice_value










