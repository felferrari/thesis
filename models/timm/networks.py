import segmentation_models_pytorch as smp
import timm
from torch import nn
import torch
from .layers import Decoder, Classifier, SkipConnConcat, JointFusionConcat
from abc import abstractclassmethod

model = timm.create_model('resnet26d', features_only = True, pretrained=False)


class GenericModel(nn.Module):
    def __init__(self, params, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.opt_input = len(params['train_opt_imgs'][0]) * params['opt_bands'] + 1
        self.sar_input = len(params['train_sar_imgs'][0]) * params['sar_bands'] + 1
        self.ef_input = self.opt_input + self.sar_input - 1

        self.n_opt_imgs = params['train_opt_imgs'][0]
        self.n_sar_imgs = params['train_sar_imgs'][0]

        self.n_classes = params['n_classes']
        #self.depths = params['resunet_depths']

    @abstractclassmethod
    def create_model(self):
        pass

class SingleInputModel(GenericModel):
    def create_model(self, in_channels, base_encoder):
        self.encoder = timm.create_model(base_encoder, features_only = True, pretrained=False, in_chans = in_channels)
        self.skip = SkipConnConcat(self.encoder.feature_info.channels())
        self.decoder = Decoder(self.skip.out_channels(), self.encoder.feature_info.channels(), self.encoder.feature_info.reduction()[0])
        self.classifier = Classifier(self.encoder.feature_info.channels()[0], self.n_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.skip(x)
        x = self.decoder(x)
        x = self.classifier(x)

        return x


class NetOpt(SingleInputModel):
    def __init__(self, params, *args, **kwargs) -> None:
        super().__init__(params, *args, **kwargs)
        self.create_model(self.opt_input, params['base_encoder'])

    def forward(self, x):
        x_img = torch.cat((x[0]), dim=1)
        x_out = torch.cat((x_img, x[2]), dim=1)
        
        return super().forward(x_out)
    
class NetSAR(SingleInputModel):
    def __init__(self, params, *args, **kwargs) -> None:
        super().__init__(params, *args, **kwargs)
        self.create_model(self.sar_input, params['base_encoder'])

    def forward(self, x):
        x_img = torch.cat((x[1]), dim=1)
        x_out = torch.cat((x_img, x[2]), dim=1)
        
        return super().forward(x_out)
    
class NetEF(SingleInputModel):
    def __init__(self, params, *args, **kwargs) -> None:
        super().__init__(params, *args, **kwargs)
        self.create_model(self.ef_input, params['base_encoder'])

    def forward(self, x):
        x_0 = torch.cat(x[0], dim=1)
        x_1 = torch.cat(x[1], dim=1)
        x_img = torch.cat((x_0, x_1), dim=1)
        x_out = torch.cat((x_img, x[2]), dim=1)
        
        return super().forward(x_out)
    

class NetJF(GenericModel):
    def __init__(self, params, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.encoder_0 = timm.create_model(params['base_encoder'], features_only = True, pretrained=False, in_chans = self.opt_input)
        self.encoder_1 = timm.create_model(params['base_encoder'], features_only = True, pretrained=False, in_chans = self.sar_input)

        self.joint_fusion = JointFusionConcat([self.encoder_0.feature_info.channels(), self.encoder_1.feature_info.channels()])

        self.skip = SkipConnConcat(self.joint_fusion.out_channels())
        self.decoder = Decoder(self.skip.out_channels(), self.encoder_0.feature_info.channels(), self.encoder_0.feature_info.reduction()[0])
        self.classifier = Classifier(self.encoder_0.feature_info.channels()[0], self.n_classes)

    def forward(self, x):
        x_0 = torch.cat(x[0], dim=1)
        x_0 = torch.cat((x_0, x[2]), dim=1)
        x_1 = torch.cat(x[1], dim=1)
        x_1 = torch.cat((x_1, x[2]), dim=1)

        x_0 = self.encoder_0(x_0)
        x_1 = self.encoder_1(x_1)
        x = self.joint_fusion([x_0, x_1])
        x = self.skip(x)
        x = self.decoder(x)
        x = self.classifier(x)

        return x
    
class NetLF(GenericModel):
    def __init__(self, params, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.encoder_0 = timm.create_model(params['base_encoder'], features_only = True, pretrained=False, in_chans = self.opt_input)
        self.encoder_1 = timm.create_model(params['base_encoder'], features_only = True, pretrained=False, in_chans = self.sar_input)

        self.skip_0 = SkipConnConcat(self.encoder_0.feature_info.channels())
        self.skip_1 = SkipConnConcat(self.encoder_1.feature_info.channels())

        self.decoder_0 = Decoder(self.skip_0.out_channels(), self.encoder_0.feature_info.channels(), self.encoder_0.feature_info.reduction()[0])
        self.decoder_1 = Decoder(self.skip_1.out_channels(), self.encoder_1.feature_info.channels(), self.encoder_1.feature_info.reduction()[0])

        self.classifier = Classifier(self.encoder_0.feature_info.channels()[0] + self.encoder_1.feature_info.channels()[0], self.n_classes)

    def forward(self, x):
        x_0 = torch.cat(x[0], dim=1)
        x_0 = torch.cat((x_0, x[2]), dim=1)
        x_1 = torch.cat(x[1], dim=1)
        x_1 = torch.cat((x_1, x[2]), dim=1)

        x_0 = self.encoder_0(x_0)
        x_1 = self.encoder_1(x_1)


        x_0 = self.skip_0(x_0)
        x_1 = self.skip_0(x_1)

        x_0 = self.decoder_0(x_0)
        x_1 = self.decoder_1(x_1)

        x = self.classifier(torch.cat((x_0, x_1), dim=1))
        return x