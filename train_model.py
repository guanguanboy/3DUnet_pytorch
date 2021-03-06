#需要参考pix2pix中GAN的训练函数，及3d unet分割中的代码

import torch
from torch import nn
import os
from model import *
from brtdataset import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import numpy as np
from tqdm.auto import tqdm


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
batch_size = 2
lr = 0.00001
device = DEVICE
display_step = 2
input_dim = 4
real_dim = 3


#创建模型及优化器
gen = D3UNet(input_dim, real_dim).to(DEVICE)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator(input_dim + real_dim, 1).to(DEVICE)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

"""
loss 函数的设计：
关于loss的设计的话。
也可以用三部分：
L1 loss，
光谱角loss

对抗生成网络的loss。BCEWithLogitsLoss
"""
# New parameters
adv_criterion = nn.BCEWithLogitsLoss() 
recon_criterion = nn.L1Loss() 
lambda_recon = 200

def get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator; takes the condition and returns potential images
        disc: the discriminator; takes images and the condition and
          returns real/fake prediction matrices
        real: the real images (e.g. maps) to be used to evaluate the reconstruction
        condition: the source images (e.g. satellite imagery) which are used to produce the real images
        adv_criterion: the adversarial loss function; takes the discriminator 
                  predictions and the true labels and returns a adversarial 
                  loss (which you aim to minimize)
        recon_criterion: the reconstruction loss function; takes the generator 
                    outputs and the real images and returns a reconstructuion 
                    loss (which you aim to minimize)
        lambda_recon: the degree to which the reconstruction loss should be weighted in the sum
    '''
    # Steps: 1) Generate the fake images, based on the conditions.
    #        2) Evaluate the fake images and the condition with the discriminator.
    #        3) Calculate the adversarial and reconstruction losses.
    #        4) Add the two losses, weighting the reconstruction loss appropriately.
    #### START CODE HERE ####
    fake = gen(condition)
    disc_fake_hat = disc(fake, condition)
    gen_adv_loss = adv_criterion(disc_fake_hat, torch.ones_like(disc_fake_hat))
    gen_rec_loss = recon_criterion(real, fake)
    gen_loss = gen_adv_loss + lambda_recon * gen_rec_loss
    #### END CODE HERE ####
    return gen_loss


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
        for condition, real in tqdm(dataloader):
            
            condition = condition.type(torch.FloatTensor)
            real = real.type(torch.FloatTensor)
            
            condition = condition.to(DEVICE)
            real = real.to(DEVICE)

            ### Update discriminator ###
            disc_opt.zero_grad() #每次循环的时候，需清空之前跑结果时所得到的梯度信息
            with torch.no_grad():
                fake = gen(condition)
            
            disc_fake_hat = disc(fake.detach(), condition) # Detach generator
            disc_fake_loss = adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))

            disc_real_hat = disc(real, condition)
            disc_real_loss = adv_criterion(disc_real_hat, torch.ones_like(disc_real_hat))

            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward(retain_graph=True) # Update gradients
            disc_opt.step() # Update optimizer
            
            ### Update generator ###
            gen_opt.zero_grad() #zero out the gradient before backpropagation
            gen_loss = get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon)
            gen_loss.backward() #Update gradients
            gen_opt.step() #Update optimizer

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step

            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step


            #Logging
            if cur_step % display_step == 0:
                if cur_step > 0:
                    print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, \
                        Discriminator loss: {mean_discriminator_loss}")
                else:
                    print("Pretrained initial state")

                mean_generator_loss = 0
                mean_discriminator_loss = 0

            #step ++,每一次循环，每一个batch的处理，叫做一个step
            cur_step += 1

        if save_model:
            torch.save({
                'gen': gen.state_dict(),
                'gen_opt': gen_opt.state_dict(),
                'disc': disc.state_dict(),
                'disc_opt': disc_opt.state_dict()
            }, f"pix2pix3d_{epoch}.pth")

train(save_model=True)
        
