from .layers import ResUnetEncoder, ResUnetDecoder, ResUnetClassifier, PrevDefPooling
import torch
from abc import abstractmethod
from einops import rearrange
from ..utils import ModelModule

class GenericModel(ModelModule):
    def __init__(self, params, training_params) -> None:
        super(GenericModel, self).__init__(training_params)
        self.opt_input = params['opt_bands'] 
        self.sar_input = params['sar_bands'] 

        #self.opt_imgs = len(params['train_opt_imgs'][0])
        #self.sar_imgs = len(params['train_sar_imgs'][0])
        self.n_classes = params['n_classes']
        self.depths = params['resunet_depths']
        self.siamese_operation = params['siamese_op']

    def get_opt(self, x):
        return x[0][:, 0], x[0][:, -1]
    
    def get_sar(self, x):
        return x[1][:, 0], x[1][:, -1]

class GenericSiamese(GenericModel):
    def prepare_model(self, in_channels):
        self.encoder = ResUnetEncoder(in_channels, self.depths)
        self.decoder = ResUnetDecoder(self.depths)
        self.classifier = ResUnetClassifier(self.depths[0], self.n_classes)

    @abstractmethod
    def prepare_input(self, x):
        pass

    def siamese_diff(self, x_0, x_1):
        return [x[1] - x[0] for x in zip(x_0, x_1)]
    
    def siamese_concat(self, x_0, x_1):
        return [torch.cat([x[1], x[0]], dim=1) for x in zip(x_0, x_1)]
    
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

class SiameseOpt(GenericSiamese):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.prepare_model(self.opt_input)

    def prepare_input(self, x):
        #x_img = torch.cat(x[0], dim=1)
        return self.get_opt(x)
    
class SiameseSAR(GenericSiamese):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.prepare_model(self.sar_input)

    def prepare_input(self, x):
        #x_img = torch.cat(x[1], dim=1)
        return self.get_sar(x)
    
class GenericSiamesePrevDef(GenericModel):
    def prepare_model(self, in_channels):
        self.encoder = ResUnetEncoder(in_channels, self.depths)
        self.decoder = ResUnetDecoder(self.depths)
        self.classifier = ResUnetClassifier(self.depths[0], self.n_classes)
        self.prev_def_poolings = PrevDefPooling(len(self.depths))

    @abstractmethod
    def prepare_input(self, x):
        pass

    def siamese_diff(self, x_0, x_1, x_2):
        return [torch.cat([x[1] - x[0], x[2]], dim = 1) for x in zip(x_0, x_1, x_2)]
    
    def siamese_concat(self, x_0, x_1, x_2):
        return [torch.cat([x[1], x[0], x[2]], dim = 1) for x in zip(x_0, x_1, x_2)]
    
    def forward(self, x):
        x = self.prepare_input(x)
        x_1 = self.encoder(x[0][1])
        x_0 = self.encoder(x[0][0])
        x_2 = self.prev_def_poolings(x[1])
        if self.siamese_operation == 'difference':
            x = self.siamese_diff(x_0, x_1, x_2)
        elif self.siamese_operation == 'concatenation':
            x = self.siamese_concat(x_0, x_1, x_2)
        x = self.decoder(x)
        x = self.classifier(x)
        return x
    
class SiameseOptPrevDef(GenericSiamesePrevDef):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.prepare_model(self.opt_input)

    def prepare_input(self, x):
        #x_img = torch.cat(x[0], dim=1)
        return self.get_opt(x), x[2]
    
class SiameseSARPrevDef(GenericSiamesePrevDef):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.prepare_model(self.sar_input)

    def prepare_input(self, x):
        #x_img = torch.cat(x[1], dim=1)
        return self.get_sar(x), x[2]