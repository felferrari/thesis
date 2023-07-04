from typing import Any
import torch
import lightning as L
from pydoc import locate
from torchmetrics.classification import MulticlassF1Score

class ModelModule(L.LightningModule):
    def __init__(self, training_params):
        super().__init__()
        self.save_hyperparameters()
        #self.class_weights = torch.tensor(training_params['loss_fn']['weights'])
        #self.loss = locate(training_params['loss_fn']['module'])(weight = torch.tensor(training_params['loss_fn']['weights']), ignore_index = training_params['loss_fn']['ignore_index'])
        self.loss = locate(training_params['loss_fn']['module'])
        self.n_classes = training_params['n_classes']
        self.loss_weights = torch.tensor(training_params['loss_fn']['weights'])
        #self.loss_ignore_index = training_params['loss_fn']['ignore_index']
        #self.train_metric = MulticlassF1Score(num_classes = training_params['n_classes'], average= 'none')
        #self.val_metric = MulticlassF1Score(num_classes = training_params['n_classes'], average= 'none')

    #def to(self, *args, **kargs):
    #    super().to(*args, **kargs)
    #    self.class_weights = self.class_weights.to(*args, **kargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        def_target = y[0]
        def_prev = self.forward(x)
        def_target_one = torch.nn.functional.one_hot(def_target, self.n_classes).moveaxis(-1, -3).float()
        loss_batch = self.loss(def_prev, def_target_one, reduction = 'mean')
        #loss_batch = self.loss(def_prev, def_target)
        #self.train_metric.to('cpu')
        #f1 = self.train_metric(def_prev.to('cpu'), def_target.to('cpu'))
        self.log("train_loss", loss_batch, prog_bar=True, logger = True, on_step=True, on_epoch=True)
        #self.log("train_f1_class_0",f1[0].item(), prog_bar=False, logger = True, on_step=True, on_epoch=True)
        #self.log("train_f1_class_1",f1[1].item(), prog_bar=False, logger = True, on_step=True, on_epoch=True)
        return loss_batch
    
    def on_train_epoch_end(self) -> None:
        #self.train_metric.reset()
        return super().on_train_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        def_target = y[0]
        def_prev = self.forward(x)
        def_target_one = torch.nn.functional.one_hot(def_target, self.n_classes).moveaxis(-1, -3).float()
        loss_batch = self.loss(def_prev, def_target_one, reduction = 'mean')
        #loss_batch = self.loss(def_prev, def_target)
        #self.val_metric.to('cpu')
        #f1 = self.val_metric(def_prev.to('cpu'), def_target.to('cpu'))
        self.log("val_loss", loss_batch, prog_bar=True, logger = True, on_step=True, on_epoch=True)
        #self.log("val_f1_class_0",f1[0].item(), prog_bar=False, logger = True, on_step=True, on_epoch=True)
        #self.log("val_f1_class_1",f1[1].item(), prog_bar=False, logger = True, on_step=True, on_epoch=True)
        return loss_batch
    
    def on_validation_epoch_end(self) -> None:
        #self.val_metric.reset()
        return super().on_validation_epoch_end()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, y = batch
        return self.forward(x)
    
