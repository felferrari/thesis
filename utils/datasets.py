from typing import Any, Optional, Sequence
import lightning.pytorch as pl
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms.functional import vflip, hflip
import torch
import random
from skimage.util import view_as_windows
import h5py
from einops import rearrange
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.core import LightningModule
from lightning.pytorch.trainer import Trainer


class GenericTrainDataset(Dataset):
    def __init__(self, params, data_folder, n_patches):
        #self.device = device
        self.params = params
        self.n_patches = n_patches
        self.data_folder = data_folder

        self.n_opt_img_groups = len(params['train_opt_imgs'])
        self.opt_imgs = params['train_opt_imgs']
        
        self.n_sar_img_groups = len(params['train_sar_imgs'])
        self.sar_imgs = params['train_sar_imgs']

    def __len__(self):
        return self.n_patches * self.n_opt_img_groups * self.n_sar_img_groups
    
    def augment_data(self, *args):
        return args
    
    
    def __getitem__(self, index):
        patch_index = index // (self.n_opt_img_groups * self.n_sar_img_groups)

        group_idx = index % (self.n_opt_img_groups * self.n_sar_img_groups)
        opt_group_index = group_idx // self.n_sar_img_groups
        sar_group_index = group_idx % self.n_sar_img_groups

        opt_images_idx = self.opt_imgs[opt_group_index]
        sar_images_idx = self.sar_imgs[sar_group_index]

        data = h5py.File(self.data_folder / f'{patch_index:d}.h5', 'r', rdcc_nbytes = 10*(1024**2))

        opt_patch = torch.from_numpy(data['opt'][()][opt_images_idx].astype(np.float32)).moveaxis(-1, -3)#.to(self.device)
        sar_patch = torch.from_numpy(data['sar'][()][sar_images_idx].astype(np.float32)).moveaxis(-1, -3)#.to(self.device)
        previous_patch = torch.from_numpy(data['previous'][()].astype(np.float32)).moveaxis(-1, -3)#.to(self.device)
        cloud_patch = torch.from_numpy(data['cloud'][()][opt_images_idx].astype(np.float32)).moveaxis(-1, -3)#.to(self.device)
        label_patch = torch.from_numpy(data['label'][()].astype(np.int64))#.to(self.device)


        opt_patch, sar_patch, previous_patch, cloud_patch, label_patch = self.augment_data(opt_patch, sar_patch, previous_patch, cloud_patch, label_patch)

        return [
            opt_patch,
            sar_patch,
            previous_patch,
        ], (label_patch, cloud_patch)

class TrainDataset(GenericTrainDataset):
    def augment_data(self, opt_patch, sar_patch, previous_patch, cloud_patch, label_patch):
        k = random.randint(0, 3)
        opt_patch = torch.rot90(opt_patch, k=k, dims=[-2, -1])
        sar_patch = torch.rot90(sar_patch, k=k, dims=[-2, -1])
        previous_patch = torch.rot90(previous_patch, k=k, dims=[-2, -1])
        cloud_patch = torch.rot90(cloud_patch, k=k, dims=[-2, -1])
        label_patch = torch.rot90(label_patch, k=k, dims=[-2, -1])

        if bool(random.getrandbits(1)):
            opt_patch = hflip(opt_patch)
            sar_patch = hflip(sar_patch)
            previous_patch = hflip(previous_patch)
            cloud_patch = hflip(cloud_patch)
            label_patch = hflip(label_patch)

        if bool(random.getrandbits(1)):
            opt_patch = vflip(opt_patch)
            sar_patch = vflip(sar_patch)
            previous_patch = vflip(previous_patch)
            cloud_patch = vflip(cloud_patch)
            label_patch = vflip(label_patch)
        
        return opt_patch, sar_patch, previous_patch, cloud_patch, label_patch

class ValDataset(GenericTrainDataset):
    pass


class PredDataset(Dataset):
    def __init__(self, 
                 opt_imgs_files,
                 sar_imgs_files,
                 previous_img_file,
                 patch_size, ) -> None:
        super().__init__()

        self.opt_imgs = [h5py.File(opt_img_file)['opt'][()].astype(np.float32) for opt_img_file in opt_imgs_files]
        self.opt_imgs = np.stack(self.opt_imgs, axis = -1)
        self.sar_imgs = [h5py.File(sar_img_file)['sar'][()].astype(np.float32) for sar_img_file in sar_imgs_files]
        self.sar_imgs = np.stack(self.sar_imgs, axis = -1)
        self.previous_img = h5py.File(previous_img_file)['previous'][()].astype(np.float32)
        self.previous_img = np.expand_dims(self.previous_img, axis = -1)
        self.original_size = self.previous_img.shape[:2]


        self.patch_size = patch_size
        img_pad_shape = ((patch_size, patch_size), (patch_size, patch_size), (0, 0), (0, 0))

        self.opt_imgs = np.pad(self.opt_imgs, img_pad_shape, 'reflect')
        self.sar_imgs = np.pad(self.sar_imgs, img_pad_shape, 'reflect')

        img_pad_shape = ((patch_size, patch_size), (patch_size, patch_size), (0, 0))

        self.previous_img = np.pad(self.previous_img, img_pad_shape, 'reflect')
        self.padded_size = self.previous_img.shape[:2]

        self.opt_imgs = rearrange(self.opt_imgs, 'h w c n -> (h w) c n')
        self.sar_imgs = rearrange(self.sar_imgs, 'h w c n -> (h w) c n')
        self.previous_img = rearrange(self.previous_img, 'h w n -> (h w) n')

    def set_overlap(self, overlap):

        idx = np.arange(self.padded_size[0] * self.padded_size[1]).reshape(self.padded_size)
        window_shape = (self.patch_size, self.patch_size)
        slide_step = int((1-overlap)*self.patch_size)
        self.idx_patches = view_as_windows(idx, window_shape, slide_step).reshape((-1, self.patch_size, self.patch_size))
    
    def get_original_size(self):
        return self.original_size
    
    def get_padded_size(self):
        return self.padded_size
    
    def __len__(self):
        return self.idx_patches.shape[0]
    
    def __getitem__(self, index):
        patch_idxs = self.idx_patches[index]

        opt_patch = self.opt_imgs[patch_idxs]
        opt_patch = torch.from_numpy(rearrange(opt_patch, 'h w c n -> n c h w'))
        sar_patch = self.sar_imgs[patch_idxs]
        sar_patch = torch.from_numpy(rearrange(sar_patch, 'h w c n -> n c h w'))
        previous_patch = self.previous_img[patch_idxs]
        previous_patch = torch.from_numpy(rearrange(previous_patch, 'h w n -> n h w'))

        return (opt_patch, sar_patch, previous_patch), patch_idxs


class ImageWriter(BasePredictionWriter):
    def __init__(self, original_size, patch_size, n_classes):
        pred_size = original_size + (n_classes, )
        self.sum_predictions = torch.zeros(pred_size, dtype=torch.float32)        
        self.count_predictions = torch.zeros(original_size, dtype = torch.int16)
        self.ones = torch.ones((patch_size, patch_size))

    def get_image(self):
        padded_img = self.sum_predictions / self.count_predictions
        return padded_img

    def write_on_batch_end(self, trainer, pl_module, prediction: Any, batch_indices, batch: Any, batch_idx, dataloader_idx):
        return super().write_on_batch_end(trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx)


