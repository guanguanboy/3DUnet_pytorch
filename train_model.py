#需要参考pix2pix中GAN的训练函数，及3d unet分割中的代码

import torch
from torch import nn
import os
from model import *
from brtdataset import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import numpy as np


#创建Dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import brtdataset
import h5py

HOME_DIR = "./BraTS-Data/"

    
#准备数据
base_dir = HOME_DIR + "processed/"

with open(base_dir + "config.json") as json_file:
    config = json.load(json_file)
    
#创建Dataset
train_dataset = brtdataset.VolumeDataSet(config["train"], base_dir + "train/", dim=(160, 160, 16), verbose=0)

steps_per_epoch = 20
n_epochs=10
batch_size = 3
initial_learning_rate = 0.00001
device = DEVICE
display_step = 20

"""
loss 函数的设计：
关于loss的设计的话。
也可以用三部分：
L1 loss，
光谱角loss

对抗生成网络的loss。BCEWithLogitsLoss
"""

def train(save_model=False):
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True)
    cur_step = 0

    for epoch in range(n_epochs):
        # Dataloader returns the batches
        for image, _ in tqdm(dataloader):
            
