import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import h5py
import json
from brtdataset import *

HOME_DIR = "./BraTS-Data/"

base_dir = HOME_DIR + "processed/"

with open(base_dir + "config.json") as json_file:
    config = json.load(json_file)
    
#创建Dataset
train_dataset = VolumeDataSet(config["train"], base_dir + "train/", dim=(160, 160, 16), verbose=0)
#创建DataLoader
dataloader = DataLoader(
    train_dataset,
    batch_size=3,
    shuffle=True)
print(len(train_dataset))
print(type(train_dataset))
#print(next(iter(dataloader)))
print(next(iter(dataloader))[0].shape) #torch.Size([3, 4, 160, 160, 16])
print(type(next(iter(dataloader))[0]))
print(next(iter(dataloader))[0].dtype)
print(next(iter(dataloader))[1].shape) #torch.Size([3, 3, 160, 160, 16])
