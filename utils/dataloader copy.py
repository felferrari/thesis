from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import vflip, hflip
import torch
import random
from utils.ops import load_sb_image, load_opt_image, load_SAR_image
from skimage.util import view_as_windows


class GenericTrainDataset(Dataset):
    def __init__(self, device, params, data_folder, n_patches):
        self.device = device
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

        opt_patch = [
            torch.from_numpy(np.load(self.data_folder / f'{self.params["prefixs"]["opt"]}_{patch_index:d}-{opt_idx}.npy').astype(np.float32)).moveaxis(2, 0).to(self.device)
            for opt_idx in opt_images_idx
            ]
        
        sar_patch = [
            torch.from_numpy(np.load(self.data_folder / f'{self.params["prefixs"]["sar"]}_{patch_index:d}-{sar_idx}.npy').astype(np.float32)).moveaxis(2, 0).to(self.device)
            for sar_idx in sar_images_idx
            ]

        previous_patch = np.load(self.data_folder / f'{self.params["prefixs"]["previous"]}_{patch_index:d}.npy').astype(np.float32)
        previous_patch = torch.from_numpy(previous_patch).moveaxis(2, 0).to(self.device)

        label_patch = np.load(self.data_folder / f'{self.params["prefixs"]["label"]}_{patch_index:d}.npy').astype(np.int64)
        label_patch = torch.from_numpy(label_patch).to(self.device)

        opt_patch, sar_patch, previous_patch, label_patch = self.augment_data(opt_patch, sar_patch, previous_patch, label_patch)

        return [
            opt_patch,
            sar_patch,
            previous_patch,
        ], label_patch

class TrainDataset(GenericTrainDataset):
    def augment_data(self, opt_patch, sar_patch, previous_patch, label_patch):
        k = random.randint(0, 3)
        opt_patch = [torch.rot90(p, k=k, dims=[1, 2]) for p in opt_patch]
        sar_patch = [torch.rot90(p, k=k, dims=[1, 2]) for p in sar_patch]
        previous_patch = torch.rot90(previous_patch, k=k, dims=[1, 2])
        label_patch = torch.rot90(label_patch, k=k, dims=[0, 1])

        if bool(random.getrandbits(1)):
            opt_patch = [hflip(p) for p in opt_patch]
            sar_patch = [hflip(p) for p in sar_patch]
            previous_patch = hflip(previous_patch)
            label_patch = hflip(label_patch)

        if bool(random.getrandbits(1)):
            opt_patch = [vflip(p) for p in opt_patch]
            sar_patch = [vflip(p) for p in sar_patch]
            previous_patch = vflip(previous_patch)
            label_patch = vflip(label_patch)
        
        return opt_patch, sar_patch, previous_patch, label_patch

class ValDataset(GenericTrainDataset):
    pass


class PredDataset(Dataset):
    def __init__(self, patch_size, device, params, data_folder) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.device = device
        self.params = params
        self.data_folder = data_folder

        previous = np.load(self.data_folder / f'{self.params["prefixs"]["previous"]}.npy').astype(np.float16)

        label = np.load(self.data_folder / f'{self.params["prefixs"]["label"]}.npy').astype(np.uint8)
        self.original_label = label

        self.original_shape = label.shape

        pad_shape = ((patch_size, patch_size),(patch_size, patch_size))
        label = np.pad(label, pad_shape, mode='reflect')
        previous = np.pad(previous, pad_shape, mode='reflect')

        self.padded_shape = label.shape

        self.label = label.flatten()
        self.previous = previous.flatten()
        #self.previous = torch.from_numpy(np.expand_dims(previous, axis=0)).to(self.device)

    def load_opt_data(self, opt_images_idx):
        pad_shape = ((self.patch_size, self.patch_size),(self.patch_size, self.patch_size), (0, 0))

        self.opt_data = [
            np.pad(np.load(self.data_folder / f'{self.params["prefixs"]["opt"]}_{opt_idx}.npy').astype(np.float16), pad_shape, mode='reflect').reshape((-1, self.params['opt_bands']))
            for opt_idx in opt_images_idx
            ]

    
    def load_sar_data(self, sar_images_idx):
        pad_shape = ((self.patch_size, self.patch_size),(self.patch_size, self.patch_size), (0, 0))
        self.sar_data = [
            np.pad(np.load(self.data_folder / f'{self.params["prefixs"]["sar"]}_{sar_idx}.npy').astype(np.float16), pad_shape, mode='reflect').reshape((-1, self.params['sar_bands']))
            for sar_idx in sar_images_idx
            ]

    def generate_overlap_patches(self, overlap):
        window_shape = (self.patch_size, self.patch_size)
        slide_step = int((1-overlap) * self.patch_size)
        idx_patches = np.arange(self.padded_shape[0]*self.padded_shape[1]).reshape(self.padded_shape)
        self.idx_patches = view_as_windows(idx_patches, window_shape, slide_step).reshape((-1, self.patch_size, self.patch_size))

    def __len__(self):
        return len(self.idx_patches)
    
    def __getitem__(self, index):
        patch_idx = self.idx_patches[index]

        opt_patch = [ 
            torch.from_numpy(np.moveaxis(p[patch_idx], 2, 0).astype(np.float32)).to(self.device)
            for p in self.opt_data 
            ]
        sar_patch = [ 
            torch.from_numpy(np.moveaxis(p[patch_idx], 2, 0).astype(np.float32)).to(self.device)
            for p in self.sar_data 
            ]
        previous_patch = np.expand_dims(self.previous[patch_idx], axis=0)
        label_patch = self.label[patch_idx]

        return (
            opt_patch,
            sar_patch,
            torch.from_numpy(previous_patch.astype(np.float32)).to(self.device)
        ), torch.from_numpy(label_patch.astype(np.int64)).to(self.device)
        


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

    




