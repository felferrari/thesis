from .layers import ResUnetEncoder, ResUnetDecoder, ResUnetClassifier, ResUnetDecoderJF, ResUnetDecoderJFNoSkip
from torch import nn
import torch

class ResUnetOpt(nn.Module):
    def __init__(self, params):
        super(ResUnetOpt, self).__init__()
        input_depth = params['params']['input_depth']
        depths = params['params']['resunet_depths']
        n_classes = params['params']['n_classes']
        self.encoder = ResUnetEncoder(input_depth, depths)
        self.decoder = ResUnetDecoder(depths)
        self.classifier = ResUnetClassifier(depths[0], n_classes)

    def forward(self, x):
        x = torch.cat((x[0], x[1], x[4]), dim=1)

        x = self.encoder(x)
        x = self.decoder(x)
        x = self.classifier(x)

        return x
    
   
class ResUnetSAR(nn.Module):
    def __init__(self, params):
        super(ResUnetSAR, self).__init__()
        input_depth = params['params']['input_depth']
        depths = params['params']['resunet_depths']
        n_classes = params['params']['n_classes']
        self.encoder = ResUnetEncoder(input_depth, depths)
        self.decoder = ResUnetDecoder(depths)
        self.classifier = ResUnetClassifier(depths[0], n_classes)

    def forward(self, x):
        x = torch.cat((x[2], x[3], x[4]), dim=1)

        x = self.encoder(x)
        x = self.decoder(x)
        x = self.classifier(x)

        return x
    

    
class ResUnetEF(nn.Module):
    def __init__(self, params):
        super(ResUnetEF, self).__init__()
        input_depth = params['params']['input_depth']
        depths = params['params']['resunet_depths']
        n_classes = params['params']['n_classes']
        self.encoder = ResUnetEncoder(input_depth, depths)
        self.decoder = ResUnetDecoder(depths)
        self.classifier = ResUnetClassifier(depths[0], n_classes)

    def forward(self, x):
        x = torch.cat((x[0], x[1], x[2], x[3], x[4]), dim=1)

        x = self.encoder(x)
        x = self.decoder(x)
        x = self.classifier(x)

        return x

class ResUnetJF(nn.Module):
    def __init__(self, input_depth_0, input_depth_1, depths, n_classes):
        super(ResUnetJF, self).__init__()
        self.encoder_0 = ResUnetEncoder(input_depth_0, depths)
        self.encoder_1 = ResUnetEncoder(input_depth_1, depths)
        self.decoder = ResUnetDecoderJF(depths)
        self.classifier = ResUnetClassifier(depths[0], n_classes)


    def forward(self, x):
        x_0 = torch.cat((x[0], x[1], x[4]), dim=1)
        x_1 = torch.cat((x[2], x[3], x[4]), dim=1)

        x_0 = self.encoder_0(x_0)
        x_1 = self.encoder_1(x_1)
        x = []
        for i in range(len(x_0)):
            x_cat = torch.cat((x_0[i], x_1[i]), dim=1)
            x.append(x_cat)

        x = self.decoder(x)
        x = self.classifier(x)

        return x
    
class ResUnetJFNoSkip(nn.Module):
    def __init__(self, input_depth_0, input_depth_1, depths, n_classes):
        super(ResUnetJFNoSkip, self).__init__()
        self.encoder_0 = ResUnetEncoder(input_depth_0, depths)
        self.encoder_1 = ResUnetEncoder(input_depth_1, depths)
        self.decoder = ResUnetDecoderJFNoSkip(depths)
        self.classifier = ResUnetClassifier(depths[0], n_classes)


    def forward(self, x):
        x_0 = torch.cat((x[0], x[1], x[4]), dim=1)
        x_1 = torch.cat((x[2], x[3], x[4]), dim=1)

        x_0 = self.encoder_0(x_0)
        x_1 = self.encoder_1(x_1)
        x = torch.cat((x_0[-1], x_1[-1]), dim=1)

        x = self.decoder(x)
        x = self.classifier(x)

        return x
    
class ResUnetLF(nn.Module):
    def __init__(self, input_depth_0, input_depth_1, depths, n_classes):
        super(ResUnetLF, self).__init__()
        self.encoder_0 = ResUnetEncoder(input_depth_0, depths)
        self.encoder_1 = ResUnetEncoder(input_depth_1, depths)
        self.decoder_0 = ResUnetDecoder(depths)
        self.decoder_1 = ResUnetDecoder(depths)
        self.classifier = ResUnetClassifier(2*depths[0], n_classes)

    def forward(self, x):
        x_0 = torch.cat((x[0], x[1], x[4]), dim=1)
        x_1 = torch.cat((x[2], x[3], x[4]), dim=1)

        x_0 = self.encoder_0(x_0)
        x_1 = self.encoder_1(x_1)

        x_0 = self.decoder_0(x_0)
        x_1 = self.decoder_1(x_1)

        x = torch.cat((x_0, x_1), dim=1)

        x = self.classifier(x)

        return x    

