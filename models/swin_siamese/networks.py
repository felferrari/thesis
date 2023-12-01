from torch import nn
import torch
from .layers import SwinEncoder, SwinDecoder, SwinClassifier, SwinDecoderJF, SwinRegressionClassifier, PrevDefPooling
from abc import abstractmethod
from einops import rearrange
from ..utils import ModelModule, ModelModuleMultiTask
#from conf import general

class GenericModel(ModelModule):
    def __init__(self, params, training_params) -> None:
        super(GenericModel, self).__init__(training_params)
        self.opt_input = len(params['train_opt_imgs'][0]) * params['opt_bands']
        self.sar_input = len(params['train_sar_imgs'][0]) * params['sar_bands']

        #self.opt_imgs = len(params['train_opt_imgs'][0])
        #self.sar_imgs = len(params['train_sar_imgs'][0])
        self.n_classes = params['n_classes']

        self.img_size = params['swin_params']['img_size']
        self.base_dim = params['swin_params']['base_dim']
        self.window_size = params['swin_params']['window_size']
        self.shift_size = params['swin_params']['shift_size']
        self.patch_size = params['swin_params']['patch_size']
        self.n_heads = params['swin_params']['n_heads']
        self.n_blocks = params['swin_params']['n_blocks']

        self.siamese_operation = params['siamese_op']

        self.encoder = SwinEncoder(
            base_dim = self.base_dim, 
            window_size = self.window_size,
            shift_size = self.shift_size,
            img_size = self.img_size,
            patch_size = self.patch_size,
            n_heads = self.n_heads,
            n_blocks = self.n_blocks
            )
        
        self.decoder = SwinDecoder(
            base_dim=self.base_dim,
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size
            )
                
        self.classifier = SwinClassifier(
            self.base_dim, 
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size,
            n_classes = self.n_classes)

    def get_opt(self, x):
        return x[0][:, 0], x[0][:, -1]
    
    def get_sar(self, x):
        return x[1][:, 0], x[1][:, -1]

    @abstractmethod
    def prepare_input(self, x):
        pass

    def siamese_diff(self, x_0, x_1):
        return [x[1] - x[0] for x in zip(x_0, x_1)]
    
    def siamese_concat(self, x_0, x_1):
        return [torch.cat([x[1], x[0]], dim=-1) for x in zip(x_0, x_1)]
    
    def forward(self, x):
        x = self.prepare_input(x)
        x_0 = self.encoder(x[0])
        x_1 = self.encoder(x[1])
        if self.siamese_operation == 'difference':
            x = self.siamese_diff(x_0, x_1)
        elif self.siamese_operation == 'concatenation':
            x = self.siamese_concat(x_0, x_1)
        x = self.decoder(x)
        x = self.classifier(x)
        return x

class SiameseOpt(GenericModel):
    def prepare_input(self, x):
        return self.get_opt(x)

class SiameseSAR(GenericModel):
    def prepare_input(self, x):
        return self.get_sar(x)
    


class GenericModelPrevDef(ModelModule):
    def __init__(self, params, training_params) -> None:
        super(GenericModelPrevDef, self).__init__(training_params)
        self.opt_input = len(params['train_opt_imgs'][0]) * params['opt_bands']
        self.sar_input = len(params['train_sar_imgs'][0]) * params['sar_bands']

        #self.opt_imgs = len(params['train_opt_imgs'][0])
        #self.sar_imgs = len(params['train_sar_imgs'][0])
        self.n_classes = params['n_classes']

        self.img_size = params['swin_params']['img_size']
        self.base_dim = params['swin_params']['base_dim']
        self.window_size = params['swin_params']['window_size']
        self.shift_size = params['swin_params']['shift_size']
        self.patch_size = params['swin_params']['patch_size']
        self.n_heads = params['swin_params']['n_heads']
        self.n_blocks = params['swin_params']['n_blocks']

        self.siamese_operation = params['siamese_op']

        self.encoder = SwinEncoder(
            base_dim = self.base_dim, 
            window_size = self.window_size,
            shift_size = self.shift_size,
            img_size = self.img_size,
            patch_size = self.patch_size,
            n_heads = self.n_heads,
            n_blocks = self.n_blocks
            )
        
        self.decoder = SwinDecoder(
            base_dim=self.base_dim,
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size
            )
                
        self.classifier = SwinClassifier(
            self.base_dim, 
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size,
            n_classes = self.n_classes)
        
        self.prev_def_poolings = PrevDefPooling(len(self.n_blocks))

    def get_opt(self, x):
        return x[0][:, 0], x[0][:, -1]
    
    def get_sar(self, x):
        return x[1][:, 0], x[1][:, -1]

    @abstractmethod
    def prepare_input(self, x):
        pass

    def siamese_diff(self, x_0, x_1, x_2):
        return [torch.cat([x[1] - x[0], x[2]], dim=-1) for x in zip(x_0, x_1, x_2)]
    
    def siamese_concat(self, x_0, x_1, x_2):
        return [torch.cat([x[1], x[0], x[2]], dim=-1) for x in zip(x_0, x_1, x_2)]
    
    def forward(self, x):
        x = self.prepare_input(x)
        x_0 = self.encoder(x[0][0])
        x_1 = self.encoder(x[0][1])
        x_2 = self.prev_def_poolings(x[1])
        if self.siamese_operation == 'difference':
            x = self.siamese_diff(x_0, x_1, x_2)
        elif self.siamese_operation == 'concatenation':
            x = self.siamese_concat(x_0, x_1, x_2)
        x = self.decoder(x)
        x = self.classifier(x)
        return x

class SiameseOptPrevDef(GenericModelPrevDef):
    def prepare_input(self, x):
        return self.get_opt(x), x[2]

class SiameseSARPrevDef(GenericModelPrevDef):
    def prepare_input(self, x):
        return self.get_sar(x), x[2]
    