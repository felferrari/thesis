from torch import nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, out_dim, stride = 1):
        super(ResidualBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(out_dim, kernel_size=3, stride = stride, padding=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(out_dim, kernel_size=3, padding=1)
        )
        self.idt_conv = nn.Sequential(
            nn.LazyConv2d(out_dim, kernel_size=3, padding=1, stride = stride),
        )
        

    def forward(self, x):
        
        return self.res_block(x) + self.idt_conv(x)

class ResUnetEncoder(nn.Module):
    def __init__(self, input_depth, depths):
        super(ResUnetEncoder, self).__init__()
        self.first_res_block = nn.Sequential(
            nn.LazyConv2d(depths[0], kernel_size=3, padding=1, padding_mode = 'reflect'),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(depths[0], kernel_size=3, padding=1, padding_mode = 'reflect')
        )
        self.first_res_cov = nn.LazyConv2d(depths[0], kernel_size=1)

        self.blocks = nn.ModuleList(
            [ResidualBlock(depths[i+1], stride = 2) for i in range(len(depths)-1)]
        )

    def forward(self, x):
        #first block
        x_idt = self.first_res_cov(x)
        x = self.first_res_block(x)
        x_0 = x + x_idt

        #encoder blocks
        x = [x_0]
        for i, block in enumerate(self.blocks):
            x.append(block(x[i]))

        return tuple(x)
    
class ResUnetDecoder(nn.Module):
    def __init__(self, depths):
        super(ResUnetDecoder, self).__init__()

        self.dec_blocks = nn.ModuleList(
            [ResidualBlock(depths[i-1]) for i in range(1, len(depths))]
        )

        self.up_blocks = nn.ModuleList(
            [nn.Upsample(scale_factor=2) for i in range(1, len(depths))]
        )


    def forward(self, x):
        x_out = x[-1]
        for i in range(len(x)-1, 0, -1):
            x_out = self.up_blocks[i-1](x_out)
            x_out = torch.cat((x_out, x[i-1]), dim=1)
            x_out = self.dec_blocks[i-1](x_out)

        return x_out  

class ResUnetDecoderJF(nn.Module):
    def __init__(self, depths):
        super(ResUnetDecoderJF, self).__init__()

        self.dec_blocks = nn.ModuleList(
            [ResidualBlock(depths[i-1]) for i in range(1, len(depths)-1)]
        )
        self.dec_blocks.append(
            ResidualBlock(depths[-2])
        )

        self.up_blocks = nn.ModuleList(
            [nn.Upsample(scale_factor=2) for i in range(1, len(depths))]
        )


    def forward(self, x):
        x_out = x[-1]
        for i in range(len(x)-1, 0, -1):
            x_out = self.up_blocks[i-1](x_out)
            x_out = torch.cat((x_out, x[i-1]), dim=1)
            x_out = self.dec_blocks[i-1](x_out)

        return x_out  



class ResUnetClassifier(nn.Module):
    def __init__(self, depth, n_classes, last_activation = nn.Softmax):
        super(ResUnetClassifier, self).__init__()
        self.res_block = ResidualBlock(depth)
        self.last_conv = nn.LazyConv2d(n_classes, kernel_size=1)
        self.last_act = last_activation(dim=1)


    def forward(self, x):
        x = self.res_block(x)
        x = self.last_conv(x)
        #x = self.last_act(x)
        return x
    
class ResUnetRegressionClassifier(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.res_block = ResidualBlock(depth)
        self.last_conv = nn.LazyConv2d(1, kernel_size=1)
        self.last_act = nn.Sigmoid()

    def forward(self, x):
        x = self.res_block(x)
        x = self.last_conv(x)
        x = self.last_act(x)
        return x

class PrevDefPooling(nn.Module):
    def __init__(self, n_depth):
        super().__init__()
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(2, 2) for _ in range(n_depth)
        ])
    
    def forward(self, x):
        output = [x]
        for pooling in self.poolings:
            x = pooling(x)
            output.append(x)

        return output