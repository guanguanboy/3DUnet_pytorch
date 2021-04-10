import torch
from torch import nn
import os
from model import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

net_input = torch.randn(1, 4, 16, 160, 160) #16为depth
#(batch_size, num_channels, depth， height, width)
net_input = net_input.to(DEVICE)
model = D3UNet(4, 3).to(DEVICE)

x = model(net_input)
print(x.shape)
#print(model)


discriminator_input_label = torch.randn(1, 3, 16, 160, 160)

discriminator_input_label = discriminator_input_label.to(DEVICE)

discriminator_input_condition = torch.randn(1, 4, 16, 160, 160)
discriminator_input_condition = discriminator_input_condition.to(DEVICE)

disc_model = Discriminator(7, 1)

disc_model = disc_model.to(DEVICE)

disc_output = disc_model(discriminator_input_label,discriminator_input_condition)

print('disc_output.shape=',disc_output.shape) #disc_output.shape= torch.Size([1, 1, 16, 160, 160])

