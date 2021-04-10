import torch
from torch import nn
import os
from model import *
from brtdataset import *

import numpy as np
from tqdm.auto import tqdm

#创建Dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import brtdataset
import h5py
from skimage.metrics import structural_similarity as cal_ssim
from skimage.metrics import peak_signal_noise_ratio as cal_psnr

HOME_DIR = "./BraTS-Data/"

    
#准备数据
base_dir = HOME_DIR + "processed/"

with open(base_dir + "config.json") as json_file:
    config = json.load(json_file)
    
steps_per_epoch = 20
n_epochs=10
batch_size = 1
lr = 0.00001
#device = DEVICE
display_step = 2
input_dim = 4
real_dim = 3



def predict():
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    #加载训练好的模型
    gen = D3UNet(input_dim, real_dim).to(device)
    gen.load_state_dict(torch.load('./pix2pix3d_9.pth')['gen'])
    gen.eval()
    gen.to(device)

    #准备数据
    #创建Dataset
    valid_dataset = brtdataset.VolumeDataSet(config["valid"], base_dir + "valid/", dim=(160, 160, 16), verbose=0)

    dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False)

    for condition, label in tqdm(dataloader):

        condition = condition.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)

        condition = condition.to(device)
        label = label.to(device)

        pred = gen(condition)

        #print('pred.shape=', pred.shape)
        #print('label.shape=', label.shape)

        #计算pnsr的值
        pred_numpy = pred.detach().cpu().numpy()
        label_numpy = label.detach().cpu().numpy()

        pred_numpy = np.squeeze(pred_numpy)
        label_numpy = np.squeeze(label_numpy)

        pred_numpy = np.moveaxis(pred_numpy, 0, -1) #将第一维，移动到最后一维
        label_numpy = np.moveaxis(label_numpy, 0, -1)
        #print(pred_numpy.shape) #(3, 160, 160, 16)
        #print('label_numpy[0].shape=', label_numpy[0].shape)
        #print('pred_numpy[0].shape=', pred_numpy[0].shape)
        pnsr = cal_psnr(label_numpy, pred_numpy)
        #print(f'pnsr={pnsr}')
        ssim = cal_ssim(label_numpy, pred_numpy, multichannel=True)
        print(f'pnsr={pnsr}, ssim = {ssim}')

if __name__=="__main__":
    predict()

