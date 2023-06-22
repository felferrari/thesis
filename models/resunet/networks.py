from .layers import ResUnetEncoder, ResUnetDecoder, ResUnetClassifier, ResUnetDecoderJF, ResUnetDecoderJFNoSkip
from torch import nn
import torch
from abc import abstractmethod
from einops import rearrange


class GenericModel(nn.Module):
    def __init__(self, params) -> None:
        super(GenericModel, self).__init__()
        self.opt_input = len(params['train_opt_imgs'][0]) * params['opt_bands'] + 1
        self.sar_input = len(params['train_sar_imgs'][0]) * params['sar_bands'] + 1

        self.opt_imgs = len(params['train_opt_imgs'][0])
        self.sar_imgs = len(params['train_sar_imgs'][0])
        self.n_classes = params['n_classes']
        self.depths = params['resunet_depths']


class GenericResunet(GenericModel):
    def prepare_model(self, in_channels):
        self.encoder = ResUnetEncoder(in_channels, self.depths)
        self.decoder = ResUnetDecoder(self.depths)
        self.classifier = ResUnetClassifier(self.depths[0], self.n_classes)

    @abstractmethod
    def prepare_input(self, x):
        pass
    
    def forward(self, x):
        x = self.prepare_input(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.classifier(x)
        return x


class ResUnetOpt(GenericResunet):
    def __init__(self, params):
        super().__init__(params)
        self.prepare_model(self.opt_input)

    def prepare_input(self, x):
        #x_img = torch.cat(x[0], dim=1)
        x_img = rearrange(x[0], 'b i c h w -> b (i c) h w')
        x = torch.cat((x_img, x[2]), dim=1)
        return x
    
   
class ResUnetSAR(GenericResunet):
    def __init__(self, params):
        super().__init__(params)
        self.prepare_model(self.sar_input)

    def prepare_input(self, x):
        #x_img = torch.cat(x[1], dim=1)
        x_img = rearrange(x[1], 'b i c h w -> b (i c) h w')
        x = torch.cat((x_img, x[2]), dim=1)
        return x
    
class ResUnetEF(GenericResunet):    
    def __init__(self, params):
        super().__init__(params)
        self.prepare_model(self.opt_input + self.sar_input - 1)

    def prepare_input(self, x):
        x_img_0 = torch.cat(x[0], dim=1)
        x_img_1 = torch.cat(x[1], dim=1)
        x = torch.cat((x_img_0, x_img_1, x[2]), dim=1)
        return x
    

class ResUnetJF(GenericModel):
    def __init__(self, params):
        super(ResUnetJF, self).__init__(params)
        self.encoder_0 = ResUnetEncoder(self.opt_input, self.depths)
        self.encoder_1 = ResUnetEncoder(self.sar_input, self.depths)
        self.decoder = ResUnetDecoderJF(self.depths)
        self.classifier = ResUnetClassifier(self.depths[0], self.n_classes)


    def forward(self, x):
        x_img_0 = torch.cat(x[0], dim=1)
        x_0 = torch.cat((x_img_0, x[2]), dim=1)

        x_img_1 = torch.cat(x[1], dim=1)
        x_1 = torch.cat((x_img_1, x[2]), dim=1)

        x_0 = self.encoder_0(x_0)
        x_1 = self.encoder_1(x_1)
        x = []
        for i in range(len(x_0)):
            x_cat = torch.cat((x_0[i], x_1[i]), dim=1)
            x.append(x_cat)

        x = self.decoder(x)
        x = self.classifier(x)

        return x
    
class ResUnetJFNoSkip(GenericModel):
    def __init__(self, params):
        super(ResUnetJFNoSkip, self).__init__(params)

        self.encoder_0 = ResUnetEncoder(self.opt_input, self.depths)
        self.encoder_1 = ResUnetEncoder(self.sar_input, self.depths)
        self.decoder = ResUnetDecoderJF(self.depths)
        self.classifier = ResUnetClassifier(self.depths[0], self.n_classes)


    def forward(self, x):
        x_img_0 = torch.cat(x[0], dim=1)
        x_0 = torch.cat((x_img_0, x[2]), dim=1)

        x_img_1 = torch.cat(x[1], dim=1)
        x_1 = torch.cat((x_img_1, x[2]), dim=1)

        x_0 = self.encoder_0(x_0)
        x_1 = self.encoder_1(x_1)
        x = torch.cat((x_0[-1], x_1[-1]), dim=1)

        x = self.decoder(x)
        x = self.classifier(x)

        return x
    
class ResUnetLF(GenericModel):
    def __init__(self, params):
        super(ResUnetLF, self).__init__(params)

        self.encoder_0 = ResUnetEncoder(self.opt_input, self.depths)
        self.encoder_1 = ResUnetEncoder(self.sar_input, self.depths)
        self.decoder_0 = ResUnetDecoder(self.depths)
        self.decoder_1 = ResUnetDecoder(self.depths)
        self.classifier = ResUnetClassifier(2*self.depths[0], self.n_classes)

    def forward(self, x):
        x_img_0 = torch.cat(x[0], dim=1)
        x_0 = torch.cat((x_img_0, x[2]), dim=1)

        x_img_1 = torch.cat(x[1], dim=1)
        x_1 = torch.cat((x_img_1, x[2]), dim=1)

        x_0 = self.encoder_0(x_0)
        x_1 = self.encoder_1(x_1)

        x_0 = self.decoder_0(x_0)
        x_1 = self.decoder_1(x_1)

        x = torch.cat((x_0, x_1), dim=1)

        x = self.classifier(x)

        return x    

