
import warnings
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_wavelets import DWT, IDWT


class WaveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet, mode='symmetric'):
        super(WaveConv2d, self).__init__()

        """
        2D Wavelet layer. It does DWT, linear transform, and Inverse dWT. 
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        size         : scalar, length of input 1D signal
        wavelet      : string, wavelet filters
        mode         : string, padding style for wavelet decomposition
        
        It initializes the kernel parameters: 
        -------------------------------------
        self.weights1 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Approximate wavelet coefficients
        self.weights2 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Horizontal-Detailed wavelet coefficients
        self.weights3 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Vertical-Detailed wavelet coefficients
        self.weights4 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Diagonal-Detailed wavelet coefficients
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if isinstance(size, list):
            if len(size) != 2:
                raise Exception('size: WaveConv2dCwt accepts the size of 2D signal in list with 2 elements')
            else:
                self.size = size
        else:
            raise Exception('size: WaveConv2dCwt accepts size of 2D signal is list')
        self.wavelet = wavelet       
        self.mode = mode
        dummy_data = torch.randn( 1,1,*self.size )        
        dwt_ = DWT(J=self.level, mode=self.mode, wave=self.wavelet)
        mode_data, mode_coef = dwt_(dummy_data)
        self.modes1 = mode_data.shape[-2]
        self.modes2 = mode_data.shape[-1]
        
        # Parameter initilization
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

    # Convolution
    def mul2d(self, input, weights):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(batch * in_channel * x * y )
                  2D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x * y)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(batch * out_channel * x * y)
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """
        Input parameters: 
        -----------------
        x : tensor, shape-[Batch * Channel * x * y]
        Output parameters: 
        ------------------
        x : tensor, shape-[Batch * Channel * x * y]
        """
        if x.shape[-1] > self.size[-1]:
            factor = int(np.log2(x.shape[-1] // self.size[-1]))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.level+factor, mode=self.mode, wave=self.wavelet).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        elif x.shape[-1] < self.size[-1]:
            factor = int(np.log2(self.size[-1] // x.shape[-1]))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.level-factor, mode=self.mode, wave=self.wavelet).to(x.device)
            x_ft, x_coeff = dwt(x)
        
        else:
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.level, mode=self.mode, wave=self.wavelet).to(x.device)
            x_ft, x_coeff = dwt(x)

        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros_like(x_ft, device= x.device)
        out_coeff = [torch.zeros_like(coeffs, device= x.device) for coeffs in x_coeff]
        
        # Multiply the final approximate Wavelet modes
        out_ft = self.mul2d(x_ft, self.weights1)
        # Multiply the final detailed wavelet coefficients
        out_coeff[-1][:,:,0,:,:] = self.mul2d(x_coeff[-1][:,:,0,:,:].clone(), self.weights2)
        out_coeff[-1][:,:,1,:,:] = self.mul2d(x_coeff[-1][:,:,1,:,:].clone(), self.weights3)
        out_coeff[-1][:,:,2,:,:] = self.mul2d(x_coeff[-1][:,:,2,:,:].clone(), self.weights4)
        
        # Return to physical space        
        idwt = IDWT(mode=self.mode, wave=self.wavelet).to(x.device)
        x = idwt((out_ft, out_coeff))
        return x

class WNO2d(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, in_channel, grid_range, padding=0):
        super(WNO2d, self).__init__()

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x,y) = g(K.v + W.v)(x,y).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : 3-channel tensor, Initial input and location (a(x,y), x,y)
              : shape: (batchsize * x=width * x=height * c=3)
        Output: Solution of a later timestep (u(x,y))
              : shape: (batchsize * x=width * x=height * c=1)
        
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        level : scalar, number of wavelet decomposition
        layers: scalar, number of wavelet kernel integral blocks
        size  : list with 2 elements (for 2D), image size
        wavelet: string, wavelet filter
        in_channel: scalar, channels in input including grid
        grid_range: list with 2 elements (for 2D), right supports of 2D domain
        padding   : scalar, size of zero padding
        """

        self.level = level
        self.width = width
        self.layers = layers
        self.size = size
        self.wavelet = wavelet
        self.in_channel = in_channel
        self.grid_range = grid_range 
        self.padding = padding
        
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        
        self.fc0 = nn.Linear(self.in_channel, self.width) # input channel is 3: (a(x, y), x, y)
        for i in range( self.layers ):
            self.conv.append( WaveConv2d(self.width, self.width, self.level, self.size, self.wavelet) )
            self.w.append( nn.Conv2d(self.width, self.width, 1) )
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)    
        x = self.fc0(x)                      # Shape: Batch * x * y * Channel
        x = x.permute(0, 3, 1, 2)            # Shape: Batch * Channel * x * y
        if self.padding != 0:
            x = F.pad(x, [0,self.padding, 0,self.padding]) 
        
        for index, (convl, wl) in enumerate( zip(self.conv, self.w) ):
            x = convl(x) + wl(x) 
            if index != self.layers - 1:     # Final layer has no activation    
                x = F.mish(x)                # Shape: Batch * Channel * x * y
                
        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]     
        x = x.permute(0, 2, 3, 1)            # Shape: Batch * x * y * Channel
        x = F.gelu( self.fc1(x) )            # Shape: Batch * x * y * Channel
        x = self.fc2(x)                      # Shape: Batch * x * y * Channel
        x = x.permute(0, 3, 1, 2)
        return x
    
    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, self.grid_range[0], size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, self.grid_range[1], size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)