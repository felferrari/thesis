from tqdm import tqdm
import torch
import os
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
from torch.nn.functional import one_hot
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import time
from pathlib import Path
from einops import rearrange

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def train_loop(dataloader, model, loss_fn, optimizer, params):
    """Executes a train loop epoch

    Args:
        dataloader (Dataloader): Pytorch Dataloader to extract train data
        model (Module): model to be trained
        loss_fn (Module): Loss Criterion
        optimizer (Optimizer): Optimizer to adjust the model's weights

    Returns:
        float: average loss of the epoch
    """
    train_loss, steps = 0, 0
    pbar = tqdm(dataloader)
    metric = MulticlassF1Score(num_classes=params['n_classes'], average = None)
    
    for (X, y) in pbar:
        optimizer.zero_grad()
        
        pred = model(X)
        loss = loss_fn(pred, y)
        steps += 1
        train_loss += loss.item()
        metric.update(pred.to('cpu'), y.to('cpu'))
        acc = metric.compute()
        pbar.set_description(f'Train Loss: {train_loss/steps:.4f}, F1-Score Classes: 0:{acc[0].item():.4f}, 1:{acc[1].item():.4f}, 2:{acc[2].item():.4f}')

        # Backpropagation
        loss.backward()
        optimizer.step()

    loss = train_loss/steps
    #print(f'Train Loss: {train_loss/steps:.4f}, Acc: 0:{acc[0].item():.4f}, 1:{acc[1].item():.4f}, 2:{acc[2].item():.4f}')
    return loss, acc[0].item(), acc[1].item()

def val_loop(dataloader, model, loss_fn, params):
    """Evaluates a validation loop epoch

    Args:
        dataloader (Dataloader): Pytorch Dataloader to extract validation data
        model (Module): model to be evaluated
        loss_fn (Module): Loss Criterion

    Returns:
        float: average loss of the epoch
    """
    val_loss, steps = 0, 0
    #f1 = MulticlassF1Score(num_classes=params['n_classes'], ignore_index = params['loss_fn']['ignore_index']).to(device)
    metric = MulticlassF1Score(num_classes=params['n_classes'], average = None)

    with torch.no_grad():
        pbar = tqdm(dataloader)
        for (X, y) in pbar:
            pred = model(X)
            loss = loss_fn(pred, y)
            steps += 1
            val_loss += loss.item()
            metric.update(pred.to('cpu'), y.to('cpu'))
            acc = metric.compute()
            pbar.set_description(f'Validation Loss: {val_loss/steps:.4f}, F1-Score Classes: 0:{acc[0].item():.4f}, 1:{acc[1].item():.4f}, 2:{acc[2].item():.4f}')

    val_loss /= steps
    #print(f'Validation Loss: {val_loss/steps:.4f}, Acc: 0:{acc[0].item():.4f}, 1:{acc[1].item():.4f}, 2:{acc[2].item():.4f}')
    return val_loss, acc[0].item(), acc[1].item()

def sample_figures_loop(dataloader, model, n_batches, epoch, path_to_samples, model_idx):

    pbar = tqdm(desc = f'Sampling figures', total = n_batches)
    for i_sample, sample in enumerate(dataloader):
        if i_sample >= n_batches:
            break
        label = sample[1]
        x = sample[0]
        pred = model(x).argmax(axis=1)
        cmap = plt.get_cmap('tab20', 3)

        plt.close('all')

        #for i, l in enumerate(label):
        figure, ax = plt.subplots(nrows=2, ncols=4, figsize = (10,5))
        p = pred[0]
        l = label[0]
        #cmap = plt.get_cmap('tab20', 1)
        img_opt_0 = rearrange(x[0][0][0].cpu().numpy(), 'c h w -> h w c')[:,:,[3,2,1]]
        img_opt_1 = rearrange(x[0][-1][0].cpu().numpy(), 'c h w -> h w c')[:,:,[3,2,1]]

        img_sar_0 = rearrange(x[1][0][0].cpu().numpy(), 'c h w -> h w c')[:,:,0]
        img_sar_1 = rearrange(x[1][-1][0].cpu().numpy(), 'c h w -> h w c')[:,:,1]

        ax[0, 0].imshow(img_opt_0*5)
        ax[0, 0].title.set_text('OPT 0')

        ax[0, 1].imshow(img_opt_1*5)
        ax[0, 1].title.set_text('OPT 1')

        ax[1, 0].imshow(img_sar_0, cmap = 'gray')
        ax[1, 0].title.set_text('SAR 0')

        ax[1, 1].imshow(img_sar_1, cmap = 'gray')
        ax[1, 1].title.set_text('SAR 1')

        l0 = ax[0, 2].imshow(l.cpu(), cmap = cmap, vmin=0, vmax = 2)
        ax[0, 2].title.set_text('Label')
        divider = make_axes_locatable(ax[0, 2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        figure.colorbar(l0, cax=cax, orientation='vertical', ticks = np.arange(3))

        l1 = ax[1, 2].imshow(l.cpu(), cmap = cmap, vmin=0, vmax = 2)
        ax[1, 2].title.set_text('Label')
        divider = make_axes_locatable(ax[1, 2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        figure.colorbar(l1, cax=cax, orientation='vertical', ticks = np.arange(3))

        p0 = ax[0, 3].imshow(p.cpu(), cmap = cmap, vmin=0, vmax = 2)
        ax[0, 3].title.set_text('Prediction')
        divider = make_axes_locatable(ax[0, 3])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        figure.colorbar(p0, cax=cax, orientation='vertical', ticks = np.arange(3))
        
        p1 = ax[1, 3].imshow(p.cpu(), cmap = cmap, vmin=0, vmax = 2)
        ax[1, 3].title.set_text('Prediction')
        divider = make_axes_locatable(ax[1, 3])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        figure.colorbar(p1, cax=cax, orientation='vertical', ticks = np.arange(3))

        plt.setp(ax, xticks=[], yticks=[])
        figure.suptitle(f'Epoch {epoch}')

        fig_path = path_to_samples / f'model_{model_idx}'
        fig_path.mkdir(exist_ok=True)
        fig_path = fig_path / f'sample_{i_sample}_{epoch}.png'
        figure.savefig(fig_path, bbox_inches='tight')
        figure.clf()
        plt.close()

        pbar.update(1)

class EarlyStop():
    def __init__(self, train_patience, path_to_save, min_delta = 0, min_epochs = None) -> None:

        self.train_pat = train_patience
        self.no_change_epochs = 0
        self.better_value = None
        self.path_to_save = path_to_save
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.decorred_epochs = 0

    def testEpoch(self, model, val_value):
        self.decorred_epochs+=1
        if self.min_epochs is not None:
            if self.decorred_epochs <= self.min_epochs:
                print(f'Epoch {self.decorred_epochs} from {self.min_epochs} minimum epochs. Validation value:{val_value:.4f}' )
                return False
        if self.better_value is None:
            self.no_change_epochs += 1
            self.better_value = val_value
            print(f'First Validation Value {val_value:.4f}. Saving model in {self.path_to_save}' )
            torch.save(model.state_dict(), self.path_to_save)
            return False
        delta = -(val_value - self.better_value)
        if delta > self.min_delta:
            self.no_change_epochs = 0
            print(f'Validation value improved from {self.better_value:.4f} to {val_value:.4f}. Saving model in {self.path_to_save}' )
            torch.save(model.state_dict(), self.path_to_save)
            self.better_value = val_value
            return False
        else:
            self.no_change_epochs += 1
            print(f'No improvement for {self.no_change_epochs}/{self.train_pat} epoch(s). Better Validation value is {self.better_value:.4f}' )
            if self.no_change_epochs >= self.train_pat:
                return True
