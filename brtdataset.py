import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import h5py
import json

class VolumeDataSet(Dataset):
    def __init__(self,
                 sample_list,
                 base_dir,
                 dim=(160, 160, 16),
                 num_channels=4,
                 num_classes=3,
                 verbose=1,
                 transform=None):
        self.base_dir = base_dir
        self.dim = dim
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.verbose = verbose
        self.sample_list = sample_list
        self.transform = transform


    def __len__(self):
        'Denotes the number of items per epoch'
        return len(self.sample_list)

    def __getitem__(self, index):
        
        #从sample_list中读取相关数据
        file_name = self.base_dir + self.sample_list[index]

        with h5py.File(file_name, 'r') as f:
            X = np.array(f.get("x"))
                # remove the background class
            y = np.moveaxis(np.array(f.get("y")), 3, 0)[1:]

        #print('X.shape=', X.shape)
        #print('y.shape=', y.shape)
        if self.transform:
            X = self.transform(X)
            y = self.transform(y)

        return X, y

#准备数据

"""
HOME_DIR = "./BraTS-Data/"

base_dir = HOME_DIR + "processed/"

with open(base_dir + "config.json") as json_file:
    config = json.load(json_file)
    
#创建Dataset
train_dataset = VolumeDataSet(config["train"], base_dir + "train/", batch_size=3, dim=(160, 160, 16), verbose=0)
#创建DataLoader
dataloader = DataLoader(
    train_dataset,
    batch_size=3,
    shuffle=True)
print(len(train_dataset))
print(type(train_dataset))
print(next(iter(dataloader)))
"""