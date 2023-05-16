from torch import nn
import torch
from .layers import SwinEncoder, SwinDecoder, SwinClassifier, SwinDecoderJF
#from conf import general

class GenericModel(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        self.opt_input = len(params['train_opt_imgs'][0]) * params['opt_bands'] + 1
        self.sar_input = len(params['train_sar_imgs'][0]) * params['sar_bands'] + 1
        self.n_classes = params['n_classes']
        self.img_size = params['swin_params']['img_size']
        self.base_dim = params['swin_params']['base_dim']
        self.window_size = params['swin_params']['window_size']
        self.shift_size = params['swin_params']['shift_size']
        self.patch_size = params['swin_params']['patch_size']
        self.n_heads = params['swin_params']['n_heads']
        self.n_blocks = params['swin_params']['n_blocks']


class SwinUnetOpt(GenericModel):
    def __init__(self, params) -> None:
        super(SwinUnetOpt, self).__init__(params)

        self.encoder = SwinEncoder(
            input_depth = self.opt_input, 
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
    
    def forward(self, x):
        x = torch.cat((x[0], x[2]), dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        #x = x.permute((0,3,1,2))
        x = self.classifier(x)
        return x
    
class SwinUnetSAR(GenericModel):
    def __init__(self, params) -> None:
        super(SwinUnetSAR, self).__init__(params)

        self.encoder = SwinEncoder(
            input_depth = self.sar_input, 
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
    
    def forward(self, x):
        x = torch.cat((x[1], x[2]), dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        #x = x.permute((0,3,1,2))
        x = self.classifier(x)
        return x

class SwinUnetEF(GenericModel):
    def __init__(self, params) -> None:
        super(SwinUnetEF, self).__init__(params)
        input_depth = self.opt_input + self.sar_input - 1
        
        self.encoder = SwinEncoder(
            input_depth = input_depth, 
            base_dim = self.base_dim, 
            window_size = self.window_size,
            shift_size = self.shift_size,
            img_size = self.img_size,
            patch_size = self.patch_size,
            n_heads =self. n_heads,
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
    
    def forward(self, x):
        x = torch.cat((x[0], x[1], x[2]), dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        #x = x.permute((0,3,1,2))
        x = self.classifier(x)
        return x

class SwinUnetJF(GenericModel):
    def __init__(self, params) -> None:
        super(SwinUnetJF, self).__init__(params)

        self.encoder_0 = SwinEncoder(
            input_depth = self.opt_input, 
            base_dim = self.base_dim, 
            window_size = self.window_size,
            shift_size = self.shift_size,
            img_size = self.img_size,
            patch_size = self.patch_size,
            n_heads = self.n_heads,
            n_blocks = self.n_blocks
            )
        
        self.encoder_1 = SwinEncoder(
            input_depth = self.sar_input, 
            base_dim = self.base_dim, 
            window_size = self.window_size,
            shift_size = self.shift_size,
            img_size = self.img_size,
            patch_size = self.patch_size,
            n_heads = self.n_heads,
            n_blocks = self.n_blocks
            )
    
        self.decoder = SwinDecoderJF(
            base_dim= self.base_dim,
            n_heads= self.n_heads,
            n_blocks = self.n_blocks,
            window_size =self. window_size,
            shift_size = self.shift_size
            )
        
        self.classifier = SwinClassifier(
            self.base_dim, 
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size,
            n_classes = self.n_classes)
    
    def forward(self, x):
        x_0 = torch.cat((x[0], x[2]), dim=1)
        x_1 = torch.cat((x[1], x[2]), dim=1)

        x_0 = self.encoder_0(x_0)
        x_1 = self.encoder_1(x_1)

        x = self.decoder([x_0, x_1])
        #x = x.permute((0,3,1,2))
        x = self.classifier(x)
        return x
    
class SwinUnetLF(GenericModel):
    def __init__(self, params) -> None:
        super(SwinUnetLF, self).__init__(params)

        self.encoder_0 = SwinEncoder(
            input_depth = self.opt_input, 
            base_dim = self.base_dim, 
            window_size = self.window_size,
            shift_size = self.shift_size,
            img_size = self.img_size,
            patch_size = self.patch_size,
            n_heads = self.n_heads,
            n_blocks = self.n_blocks
            )
        
        self.encoder_1 = SwinEncoder(
            input_depth = self.sar_input, 
            base_dim = self.base_dim, 
            window_size = self.window_size,
            shift_size = self.shift_size,
            img_size = self.img_size,
            patch_size = self.patch_size,
            n_heads = self.n_heads,
            n_blocks = self.n_blocks
            )
    
        self.decoder_0 = SwinDecoder(
            base_dim=self.base_dim,
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size
            )
        
        self.decoder_1 = SwinDecoder(
            base_dim=self.base_dim,
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size
            )
        
        self.classifier = SwinClassifier(
            2*self.base_dim, 
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size,
            n_classes =self.n_classes)
    
    def forward(self, x):
        x_0 = torch.cat((x[0], x[2]), dim=1)
        x_1 = torch.cat((x[1], x[2]), dim=1)

        x_0 = self.encoder_0(x_0)
        x_1 = self.encoder_1(x_1)

        x_0 = self.decoder_0(x_0)
        x_1 = self.decoder_1(x_1)

        x = torch.cat((x_0, x_1), dim=-1)
        x = self.classifier(x)
        return x