from tqdm import tqdm
import torch
import os
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
from torch.nn.functional import one_hot
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import time

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
    metric = MulticlassAccuracy(num_classes=params['n_classes'], average = None)
    
    for (X, y) in pbar:
        pred = model(X)
        loss = loss_fn(pred, y)
        steps += 1
        train_loss += loss.item()
        metric.update(pred.to('cpu'), y.to('cpu'))
        acc = metric.compute()
        pbar.set_description(f'Train Loss: {train_loss/steps:.4f}, Acc Classes: 0:{acc[0].item():.4f}, 1:{acc[1].item():.4f}, 2:{acc[2].item():.4f}')

        # Backpropagation
        optimizer.zero_grad()
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
    metric = MulticlassAccuracy(num_classes=params['n_classes'], average = None)
    with torch.no_grad():
        pbar = tqdm(dataloader)
        for (X, y) in pbar:
            pred = model(X)
            loss = loss_fn(pred, y)
            steps += 1
            val_loss += loss.item()
            metric.update(pred.to('cpu'), y.to('cpu'))
            acc = metric.compute()
            pbar.set_description(f'Validation Loss: {val_loss/steps:.4f}, Acc Classes: 0:{acc[0].item():.4f}, 1:{acc[1].item():.4f}, 2:{acc[2].item():.4f}')

    val_loss /= steps
    #print(f'Validation Loss: {val_loss/steps:.4f}, Acc: 0:{acc[0].item():.4f}, 1:{acc[1].item():.4f}, 2:{acc[2].item():.4f}')
    return val_loss, acc[0].item(), acc[1].item()

def val_sample_image(dataloader, model, path_to_samples, epoch):
    sample = next(iter(dataloader))
    label = sample[1]
    x = sample[0]
    pred = model(x).argmax(axis=1)
    plt.close('all')
    for i, l in enumerate(label):
        figure, ax = plt.subplots(nrows=1, ncols=2, figsize = (10,5))
        p = pred[i]
        cmap = plt.get_cmap('tab20', 10)
        im0 = ax[0].imshow(l.cpu(), cmap = cmap, vmin=-0.5, vmax = 9.5)
        ax[0].title.set_text('Label')
        im1 = ax[1].imshow(p.cpu(), cmap = cmap, vmin=-0.5, vmax = 9.5)
        ax[1].title.set_text(f'Prediction Epoch {epoch+1:03d}')
        
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        figure.colorbar(im1, cax=cax, orientation='vertical', ticks = np.arange(10))

        figure.savefig(os.path.join(path_to_samples, f'sample_{i}_{epoch}.png'), bbox_inches='tight')
        figure.clf()
        plt.close()

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
            if self.no_change_epochs > self.train_pat:
                return True
