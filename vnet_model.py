import torch
import torch.nn as nn
import torch.nn.functional as F

###### 
## PyTorch Vnet adapted from https://github.com/mattmacy/vnet.pytorch by davidvp12@gmail.com
## List of modifications:
##      - Added parametrizations for input_channels and split_channels
##      - Variable channels functionality
##      - 3D SqueezeExcitation added
##      - Added ContInstanceNorm3d (now default), ContGroupNorm3d
##      - Added SiLU activation function (now default), GELU
###### 

#from torchvision.ops import SqueezeExcitation 
# Default SE from torchvision is in 2D so custom 3D SE implemented
class SqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (torch.nn.Module, optional): Default: ``torch.nn.SiLU``
        scale_activation (torch.nn.Module): Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int = 16,
        activation: torch.nn.Module = torch.nn.SiLU,
        scale_activation: torch.nn.Module = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool3d(1)
        self.fc1 = torch.nn.Conv3d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv3d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scale = self._scale(input)
        return scale * input


def passthrough(x, **kwargs):
    return x

    
def ELUCons(elu, nchan):
    if elu:
        #return nn.ELU(inplace=True)
        #return nn.GELU(inplace=True)
        return nn.SiLU(inplace=True)
    else:
        return nn.PReLU(nchan)


# NORMALIZATION classes
# normalization between sub-volumes is necessary for good performance
# different options, batch and instance norm are prefered

# Group normalization
# class ContGroupNorm3d(nn.GroupNorm):
#     def __init__(self, num_channels, num_groups=32, enable_gn=True):
#         super(ContGroupNorm3d, self).__init__(num_channels, num_groups)
#         self.enable_gn = enable_gn
#     def forward(self, input):
#         if self.enable_gn:
#             return super(ContGroupNorm3d, self).forward(input)
#         else:
#             return input

# Instance normalization - DEFAULT
class ContInstanceNorm3d(nn.InstanceNorm3d):
    def __init__(self, num_features, enable_in=True, **kwargs):
        super(ContInstanceNorm3d, self).__init__(num_features, **kwargs)
        self.enable_in = enable_in
    def forward(self, input):
        if self.enable_in:
            return super(ContInstanceNorm3d, self).forward(input)
        else:
            return input

# Batch normalization
# class ContBatchNorm3d(nn.BatchNorm3d):
#     def __init__(self, num_features, enable_bn=True, **kwargs):
#         super(ContBatchNorm3d, self).__init__(num_features, **kwargs)
#         self.enable_bn = enable_bn

#     def forward(self, input):
#         if self.enable_bn: #and self.training:
#             return super(ContBatchNorm3d, self).forward(input)
#         else:
#             return input
        
# Convolutional module used in the up and down transitions
class ConvModule(nn.Module):
    def __init__(self, nchan, elu, se):
        super(ConvModule, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        #self.bn1 = ContBatchNorm3d(nchan)
        self.bn1 = ContInstanceNorm3d(nchan)
        self.se1 = passthrough
        if se:
            self.se1 = SqueezeExcitation(input_channels=nchan)  
        
    def forward(self, x):
        return self.relu1(self.bn1(self.conv1(self.se1(x))))
        

def _make_nConv(nchan, depth, elu, se):
    layers = []
    for _ in range(depth):
        layers.append(ConvModule(nchan, elu, se))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, input_ch, output_ch, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(input_ch, output_ch, kernel_size=5, padding=2)
        self.bn1 = ContInstanceNorm3d(output_ch)
        self.relu1 = ELUCons(elu, output_ch)

        assert output_ch % input_ch == 0, f"Error: split_ch must be a multiple of output_ch."
        self.input_ch = input_ch
        self.output_ch = output_ch

    def forward(self, x):
        # we may add elu here as well
        out = self.bn1(self.conv1(x))
        
        # split input
        num_concat = int(self.output_ch / self.input_ch) - 1
        x_split = x
        for _ in range(num_concat):
            x_split = torch.cat((x_split, x), 1)
            
        out = self.relu1(torch.add(out, x_split))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, se, dropout=True):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContInstanceNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d(p=0.2)
        self.ops = _make_nConv(outChans, nConvs, elu, se)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, se, dropout=True):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContInstanceNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d(p=0.2)
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d(p=0.2)
        self.ops = _make_nConv(outChans, nConvs, elu, se)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        self.bn1 = ContInstanceNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.relu1 = ELUCons(elu, 2)
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        
        # make channels the last axis
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        
        # flatten
        out = out.view(out.numel() // 2, 2)        
        out = self.softmax(out, dim=1)
        # treat channel 1 as the predicted output 
        # out = out[:,1]
        return out

        
class VNet(nn.Module):
    def __init__(self, elu=False, se=True, nll=False, input_ch=1, split_ch=16):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(input_ch, split_ch, elu)
        self.down_tr32 = DownTransition(split_ch, 1, elu, se)
        self.down_tr64 = DownTransition(split_ch*2, 2, elu, se)
        self.down_tr128 = DownTransition(split_ch*4, 3, elu, se, dropout=True)
        self.down_tr256 = DownTransition(split_ch*8, 2, elu, se, dropout=True)
        self.up_tr256 = UpTransition(split_ch*16, split_ch*16, 2, elu, se, dropout=True)
        self.up_tr128 = UpTransition(split_ch*16, split_ch*8, 2, elu, se, dropout=True)
        self.up_tr64 = UpTransition(split_ch*8, split_ch*4, 1, elu, se)
        self.up_tr32 = UpTransition(split_ch*4, split_ch*2, 1, elu, se)
        self.out_tr = OutputTransition(split_ch*2, elu, nll)
        
    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out
