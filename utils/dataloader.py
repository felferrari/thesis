from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import vflip, hflip
import torch
import random
from utils.ops import load_sb_image, load_opt_image, load_SAR_image
from skimage.util import view_as_windows


class GenericDataset(Dataset):
    def __init__(self, dataset_csv, patch_size, device, params, n_patches_tile = 1):
        self.data_df = pd.read_csv(dataset_csv)
        self.device = device
        self.patch_size = patch_size
        self.params = params
        self.n_patches_tile = n_patches_tile


    def __len__(self):
        return len(self.data_df)*self.n_patches_tile
    
    def augment_data(self, data, label):
        return data, label
    
    def split_data(self, data):
        opt_bands = self.params['opt_bands']
        sar_bands = self.params['sar_bands']
        b1 = opt_bands
        b2 = b1 + opt_bands
        b3 = b2 + sar_bands
        b4 = b3 + sar_bands
        b5 = b4 + 1
        return (
            data[:b1],
            data[b1:b2],
            data[b2:b3],
            data[b3:b4],
            data[b4:b5],
        )
    
    def __getitem__(self, index):
        index = index // self.n_patches_tile
        delta_size = self.params['tile_size'] - self.params['patch_size']
        d0 = random.randrange(delta_size)
        d1 = random.randrange(delta_size)
        data_file, label_file = self.data_df['data'][index], self.data_df['label'][index]
        data = np.load(data_file)[d0:d0+self.params['patch_size'], d1:d1+self.params['patch_size']].astype(np.float32)
        label = np.load(label_file)[d0:d0+self.params['patch_size'], d1:d1+self.params['patch_size']].astype(np.int64)
        if np.any(np.isnan(data)):
            print(data)

        
        data = torch.from_numpy(data).moveaxis(2, 0).to(self.device)
        label = torch.from_numpy(label).to(self.device)
        data, label = self.augment_data(data, label)

        return self.split_data(data), label

class TrainDataset(GenericDataset):
    def augment_data(self, data, label):
        k = random.randint(0, 3)
        data = torch.rot90(data, k=k, dims=[1,2])
        label = torch.rot90(label, k=k, dims=[0,1])

        if bool(random.getrandbits(1)):
            data = hflip(data)
            label = hflip(label)

        if bool(random.getrandbits(1)):
            data = vflip(data)
            label = vflip(label)
        
        return data, label

class ValDataset(GenericDataset):
    pass
        

class PredDataSet(Dataset):
    def __init__(self, images, device, patch_size, stats, transformer = ToTensor()) -> None:
        self.device = device
        self.patch_size = patch_size
        self.transformer = transformer

        opt_mean = np.array(stats['opt_mean'])
        opt_std = np.array(stats['opt_std'])
        sar_mean = np.array(stats['sar_mean'])
        sar_std = np.array(stats['sar_std'])

        self.label = load_sb_image(images['label'])

        pad_shape = ((patch_size, patch_size),(patch_size, patch_size))

        prev_def = load_sb_image(images['previous'])
        self.original_shape = prev_def.shape
        prev_def = np.pad(prev_def, pad_shape, mode = 'reflect')
        self.padded_shape = prev_def.shape[:2]
        self.prev_def = prev_def.reshape((-1, 1))

        pad_shape = ((patch_size, patch_size),(patch_size, patch_size),(0,0))

        img = load_opt_image(images['opt_0'])
        img = (img - opt_mean)/opt_std
        img = np.pad(img, pad_shape, mode = 'reflect')
        self.opt_img_0 = img.reshape((-1, img.shape[-1]))

        img = load_opt_image(images['opt_1'])
        img = (img - opt_mean)/opt_std
        img = np.pad(img, pad_shape, mode = 'reflect')
        self.opt_img_1 = img.reshape((-1, img.shape[-1]))
        
        img = load_SAR_image(images['sar_0'])
        img = (img - sar_mean)/sar_std
        img = np.pad(img, pad_shape, mode = 'reflect')
        self.sar_img_0 = img.reshape((-1, img.shape[-1]))

        img = load_SAR_image(images['sar_1'])
        img = (img - sar_mean)/sar_std
        img = np.pad(img, pad_shape, mode = 'reflect')
        self.sar_img_1 = img.reshape((-1, img.shape[-1]))



    def gen_patches(self, overlap):
        idx_patches = np.arange(self.padded_shape[0]*self.padded_shape[1]).reshape(self.padded_shape)
        slide_step = int((1-overlap)*self.patch_size)
        window_shape = (self.patch_size, self.patch_size)
        self.idx_patches = view_as_windows(idx_patches, window_shape, slide_step).reshape((-1, self.patch_size, self.patch_size))

    def __len__(self):
        return self.idx_patches.shape[0]

    def __getitem__(self, index):
        patch = self.idx_patches[index]

        opt_0 = self.transformer(self.opt_img_0[patch].astype(np.float32)).to(self.device)
        opt_1 = self.transformer(self.opt_img_1[patch].astype(np.float32)).to(self.device)

        sar_0 = self.transformer(self.sar_img_0[patch].astype(np.float32)).to(self.device)
        sar_1 = self.transformer(self.sar_img_1[patch].astype(np.float32)).to(self.device)

        prev_def = self.transformer(self.prev_def[patch].astype(np.float32)).to(self.device)

        return (
            opt_0,
            opt_1,
            sar_0,
            sar_1,
            prev_def
        )    

    




