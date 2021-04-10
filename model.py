import torch
from torch import nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ContractingBlock(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, use_maxpooling=True):
        super(ContractingBlock, self).__init__()

        self.conv3d_1 = nn.Conv3d(in_channels=input_channels, out_channels=hidden_channels, \
            kernel_size=(3, 3, 3),stride=(1, 1, 1), padding=(1, 1, 1))
        self.activation = nn.ReLU()
        self.conv3d_2 = nn.Conv3d(in_channels=hidden_channels, out_channels=output_channels, \
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.max_pooling3d = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.use_maxpool = use_maxpooling

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.activation(x)
        x = self.conv3d_2(x)
        x = self.activation(x)
        #if self.use_maxpool:
            #x = self.max_pooling3d(x)

        return x


class ExpandingBlock(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, contain_2Conv=True, isFinalLayer=False):
        super(ExpandingBlock, self).__init__()

        self.conv3d_1 = nn.Conv3d(in_channels=input_channels, out_channels=hidden_channels, \
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(in_channels=hidden_channels, out_channels=output_channels, \
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.activation = nn.ReLU()
        self.contain_2conv = contain_2Conv
        self.isFinalLayer = isFinalLayer
        self.activation_sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv3d_1(x)

        if self.isFinalLayer:
            x = self.activation_sigmoid(x)
        else:
            x = self.activation(x)

        if self.contain_2conv:
            x = self.conv3d_2(x)
            x = self.activation(x)

        return x



class D3UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(D3UNet, self).__init__()

        self.contract1 = ContractingBlock(input_channels, 32, 64, use_maxpooling=True)
        self.contract2 = ContractingBlock(64, 64, 128, use_maxpooling=False)
        self.expand1 = ExpandingBlock(192, 64, 64, contain_2Conv=True)
        self.expand2 = ExpandingBlock(64, output_channels, output_channels, contain_2Conv=False, isFinalLayer=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear') #该函数可以处理3D tensor
        self.max_pooling3d = nn.MaxPool3d(kernel_size=(2, 2, 2))

    def forward(self, x):
        x0 = self.contract1(x)
        #print('x0.shape:', x0.shape) #torch.Size([1, 64, 16, 160, 160])

        x1 = self.max_pooling3d(x0) #torch.Size([1, 64, 8, 80, 80])
        #print('x1.shape:', x1.shape)

        x2 = self.contract2(x1) #
        #print('x2.shape:', x2.shape) #x2.shape: torch.Size([1, 128, 8, 80, 80])
        
        x3 = self.upsample(x2)
        #print('x3.shape:', x3.shape) #torch.Size([1, 128, 16, 160, 160])
        #这里需要concatnate
        x4 = torch.cat([x3, x0], axis=1)
        #print('x4.shape:', x4.shape) #torch.Size([1, 192, 16, 160, 160])
        
        x5 = self.expand1(x4)
        #print('x5.shape:', x5.shape) #x5.shape: torch.Size([1, 64, 16, 160, 160])

        x6 = self.expand2(x5)
        #print('x6.shape:', x6.shape) #x6.shape: torch.Size([1, 3, 16, 160, 160])

        return x6

"""
net_input = torch.randn(1, 4, 16, 160, 160) #16为depth
#(batch_size, num_channels, depth， height, width)
net_input = net_input.to(DEVICE)
model = D3UNet(4, 3).to(DEVICE)

x = model(net_input)
print(x.shape)
print(model)
"""

"""
    FeatureMapBlock Class
    The final layer of a U-Net - 
    maps each pixel to a pixel with the correct number of output dimensions
    using a 1x1 convolution.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output

    总结起来feature map类的作用就是通过1x1的卷积变换通道的数目，
    当通道数不相同时，可以使用此函数，
    但是此函数实现的时候，使用的是2D卷积，因此只能处理类似二维的图像数据，对于3维数据
    的处理，最好使用三维卷积
"""
class FeatureMap3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureMap3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class Discriminator(nn.Module):
    '''
    Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake. 
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    '''
    def __init__(self, input_channels, output_channels):
        super(Discriminator, self).__init__()
        self.upfeature = FeatureMap3DBlock(input_channels, 16)
        self.contract1 = ContractingBlock(16, 32, 64, use_maxpooling=True)
        self.contract2 = ContractingBlock(64, 64, 128, use_maxpooling=False)
        self.final = nn.Conv3d(128, 1, kernel_size=1) #使用1x1卷积将通道数变换为1
        
    def forward(self, x, y): #x为label或fake, y为condition

        x = torch.cat([x, y], axis=1) #在通道维进行连接
        x = self.upfeature(x) #变换通道数，然后再进一步处理
        x0 = self.contract1(x)
        x1 = self.contract2(x0)
        x2 = self.final(x1)

        return x2