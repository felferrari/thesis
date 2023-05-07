from torch import nn
import torch
from .layers import SwinEncoder, SwinDecoder, SwinClassifier, SwinDecoderJF
#from conf import general

class SwinUnetOpt(nn.Module):
    def __init__(self, params) -> None:
        super(SwinUnetOpt, self).__init__()
        input_depth = 2*params['opt_bands'] + 1 
        img_size = params['img_size']
        base_dim = params['swin_base_dim']
        window_size = params['swin_window_size']
        shift_size = params['swin_shift_size']
        patch_size = params['swin_patch_size']
        n_heads = params['swin_n_heads']
        n_blocks = params['swin_n_blocks']
        n_classes = params['n_classes']

        self.encoder = SwinEncoder(
            input_depth = input_depth, 
            base_dim = base_dim, 
            window_size = window_size,
            shift_size = shift_size,
            img_size = img_size,
            patch_size = patch_size,
            n_heads = n_heads,
            n_blocks = n_blocks
            )
    
        self.decoder = SwinDecoder(
            base_dim=base_dim,
            n_heads=n_heads,
            n_blocks = n_blocks,
            window_size = window_size,
            shift_size = shift_size
            )
        
        self.classifier = SwinClassifier(
            base_dim, 
            n_heads=n_heads,
            n_blocks = n_blocks,
            window_size = window_size,
            shift_size = shift_size,
            n_classes = n_classes)
    
    def forward(self, x):
        x = torch.cat((x[0], x[1], x[4]), dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        #x = x.permute((0,3,1,2))
        x = self.classifier(x)
        return x
    
class SwinUnetSAR(nn.Module):
    def __init__(self, params) -> None:
        super(SwinUnetSAR, self).__init__()
        input_depth = 2*params['sar_bands'] + 1 
        img_size = params['img_size']
        base_dim = params['swin_base_dim']
        window_size = params['swin_window_size']
        shift_size = params['swin_shift_size']
        patch_size = params['swin_patch_size']
        n_heads = params['swin_n_heads']
        n_blocks = params['swin_n_blocks']
        n_classes = params['n_classes']
        self.encoder = SwinEncoder(
            input_depth = input_depth, 
            base_dim = base_dim, 
            window_size = window_size,
            shift_size = shift_size,
            img_size = img_size,
            patch_size = patch_size,
            n_heads = n_heads,
            n_blocks = n_blocks
            )
    
        self.decoder = SwinDecoder(
            base_dim=base_dim,
            n_heads=n_heads,
            n_blocks = n_blocks,
            window_size = window_size,
            shift_size = shift_size
            )
        
        self.classifier = SwinClassifier(
            base_dim, 
            n_heads=n_heads,
            n_blocks = n_blocks,
            window_size = window_size,
            shift_size = shift_size,
            n_classes = n_classes)
    
    def forward(self, x):
        x = torch.cat((x[2], x[3], x[4]), dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        #x = x.permute((0,3,1,2))
        x = self.classifier(x)
        return x

class SwinUnetEF(nn.Module):
    def __init__(self, params) -> None:
        super(SwinUnetEF, self).__init__()
        input_depth = 2*params['opt_bands'] + 2*params['sar_bands'] + 1 
        img_size = params['img_size']
        base_dim = params['swin_base_dim']
        window_size = params['swin_window_size']
        shift_size = params['swin_shift_size']
        patch_size = params['swin_patch_size']
        n_heads = params['swin_n_heads']
        n_blocks = params['swin_n_blocks']
        n_classes = params['n_classes']
        self.encoder = SwinEncoder(
            input_depth = input_depth, 
            base_dim = base_dim, 
            window_size = window_size,
            shift_size = shift_size,
            img_size = img_size,
            patch_size = patch_size,
            n_heads = n_heads,
            n_blocks = n_blocks
            )
    
        self.decoder = SwinDecoder(
            base_dim=base_dim,
            n_heads=n_heads,
            n_blocks = n_blocks,
            window_size = window_size,
            shift_size = shift_size
            )
        
        self.classifier = SwinClassifier(
            base_dim, 
            n_heads=n_heads,
            n_blocks = n_blocks,
            window_size = window_size,
            shift_size = shift_size,
            n_classes = n_classes)
    
    def forward(self, x):
        x = torch.cat((x[0], x[1], x[2], x[3], x[4]), dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        #x = x.permute((0,3,1,2))
        x = self.classifier(x)
        return x

class SwinUnetJF(nn.Module):
    def __init__(self, params) -> None:
        super(SwinUnetJF, self).__init__()
        input_depth_0 = 2*params['opt_bands'] + 1 
        input_depth_1 = 2*params['sar_bands'] + 1 
        img_size = params['img_size']
        base_dim = params['swin_base_dim']
        window_size = params['swin_window_size']
        shift_size = params['swin_shift_size']
        patch_size = params['swin_patch_size']
        n_heads = params['swin_n_heads']
        n_blocks = params['swin_n_blocks']
        n_classes = params['n_classes']
        self.encoder_0 = SwinEncoder(
            input_depth = input_depth_0, 
            base_dim = base_dim, 
            window_size = window_size,
            shift_size = shift_size,
            img_size = img_size,
            patch_size = patch_size,
            n_heads = n_heads,
            n_blocks = n_blocks
            )
        
        self.encoder_1 = SwinEncoder(
            input_depth = input_depth_1, 
            base_dim = base_dim, 
            window_size = window_size,
            shift_size = shift_size,
            img_size = img_size,
            patch_size = patch_size,
            n_heads = n_heads,
            n_blocks = n_blocks
            )
    
        self.decoder = SwinDecoderJF(
            base_dim=base_dim,
            n_heads=n_heads,
            n_blocks = n_blocks,
            window_size = window_size,
            shift_size = shift_size
            )
        
        self.classifier = SwinClassifier(
            base_dim, 
            n_heads=n_heads,
            n_blocks = n_blocks,
            window_size = window_size,
            shift_size = shift_size,
            n_classes = n_classes)
    
    def forward(self, x):
        x_0 = torch.cat((x[0], x[1], x[4]), dim=1)
        x_1 = torch.cat((x[2], x[3], x[4]), dim=1)

        x_0 = self.encoder_0(x_0)
        x_1 = self.encoder_1(x_1)

        x = self.decoder([x_0, x_1])
        #x = x.permute((0,3,1,2))
        x = self.classifier(x)
        return x
    
class SwinUnetLF(nn.Module):
    def __init__(self, params) -> None:
        super(SwinUnetLF, self).__init__()
        input_depth_0 = 2*params['opt_bands'] + 1 
        input_depth_1 = 2*params['sar_bands'] + 1 
        img_size = params['img_size']
        base_dim = params['swin_base_dim']
        window_size = params['swin_window_size']
        shift_size = params['swin_shift_size']
        patch_size = params['swin_patch_size']
        n_heads = params['swin_n_heads']
        n_blocks = params['swin_n_blocks']
        n_classes = params['n_classes']
        self.encoder_0 = SwinEncoder(
            input_depth = input_depth_0, 
            base_dim = base_dim, 
            window_size = window_size,
            shift_size = shift_size,
            img_size = img_size,
            patch_size = patch_size,
            n_heads = n_heads,
            n_blocks = n_blocks
            )
        
        self.encoder_1 = SwinEncoder(
            input_depth = input_depth_1, 
            base_dim = base_dim, 
            window_size = window_size,
            shift_size = shift_size,
            img_size = img_size,
            patch_size = patch_size,
            n_heads = n_heads,
            n_blocks = n_blocks
            )
    
        self.decoder_0 = SwinDecoder(
            base_dim=base_dim,
            n_heads=n_heads,
            n_blocks = n_blocks,
            window_size = window_size,
            shift_size = shift_size
            )
        
        self.decoder_1 = SwinDecoder(
            base_dim=base_dim,
            n_heads=n_heads,
            n_blocks = n_blocks,
            window_size = window_size,
            shift_size = shift_size
            )
        
        self.classifier = SwinClassifier(
            2*base_dim, 
            n_heads=n_heads,
            n_blocks = n_blocks,
            window_size = window_size,
            shift_size = shift_size,
            n_classes = n_classes)
    
    def forward(self, x):
        x_0 = torch.cat((x[0], x[1], x[4]), dim=1)
        x_1 = torch.cat((x[2], x[3], x[4]), dim=1)

        x_0 = self.encoder_0(x_0)
        x_1 = self.encoder_1(x_1)

        x_0 = self.decoder_0(x_0)
        x_1 = self.decoder_1(x_1)

        x = torch.cat((x_0, x_1), dim=-1)
        x = self.classifier(x)
        return x