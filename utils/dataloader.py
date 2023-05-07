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
    def __init__(self, dataset_csv, patch_size, device, params):
        self.data_df = pd.read_csv(dataset_csv)
        #self.data_df = self.data_df.sample(frac=1).reset_index(drop=True)
        self.device = device
        self.patch_size = patch_size
        self.params = params


    def __len__(self):
        return len(self.data_df)
    
    def augment_data(self, data, label):
        return data, label
    
    def split_data(self, data):
        opt_bands = self.params['opt_bands']*self.params['n_opt_images']
        sar_bands = self.params['sar_bands']*self.params['n_sar_images']
        b1 = opt_bands
        b2 = b1 + sar_bands
        b3 = b2 + 1
        return (
            data[:b1],
            data[b1:b2],
            data[b2:b3]
        )
    
    def __getitem__(self, index):
        data_file, label_file = self.data_df['data'][index], self.data_df['label'][index]
        data = np.load(data_file).astype(np.float32)
        label = np.load(label_file).astype(np.int64)
        
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

class PredDataSet(GenericDataset):
    pass
        

class PredDataSet2(Dataset):
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

        data_opt = []
        for img_file_i in images['opt_files']:
            img_path = images['opt_folder'] / img_file_i
            img = load_opt_image(img_path).astype(np.float16)
            img = (img - opt_mean)/opt_std
            data_opt.append(img)
        data_opt = np.concatenate(data_opt, axis=-1)
        data_opt = np.pad(data_opt, pad_shape, mode = 'reflect')
        self.opt = data_opt.reshape((-1, data_opt.shape[-1]))
        del data_opt

        data_sar = []
        for img_file_i in images['sar_files']:
            img_path = images['sar_folder'] / img_file_i
            img = load_SAR_image(img_path).astype(np.float16)
            img = (img - sar_mean)/sar_std
            data_sar.append(img)
        data_sar = np.concatenate(data_sar, axis=-1)
        data_sar = np.pad(data_sar, pad_shape, mode = 'reflect')
        self.sar = data_sar.reshape((-1, data_sar.shape[-1]))
        del data_sar


    def gen_patches(self, overlap):
        idx_patches = np.arange(self.padded_shape[0]*self.padded_shape[1]).reshape(self.padded_shape)
        slide_step = int((1-overlap)*self.patch_size)
        window_shape = (self.patch_size, self.patch_size)
        self.idx_patches = view_as_windows(idx_patches, window_shape, slide_step).reshape((-1, self.patch_size, self.patch_size))

    def __len__(self):
        return self.idx_patches.shape[0]

    def __getitem__(self, index):
        patch = self.idx_patches[index]

        opt = self.transformer(self.opt[patch].astype(np.float32)).to(self.device)

        sar = self.transformer(self.sar[patch].astype(np.float32)).to(self.device)

        prev_def = self.transformer(self.prev_def[patch].astype(np.float32)).to(self.device)

        return (
            opt,
            sar,
            prev_def
        )    

    




