# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 15:23:46 2021

@author: Karan Chilwal
"""
import pandas as pd
import os
from PIL import Image
import random

train_df = pd.DataFrame(columns = ["input_name", "label_name", "frequency", "forw_back", "reynolds"])
print(train_df)
train_df["input_name"] = os.listdir("C:/Users/Karan Chilwal/Desktop/Cropped_input")    #change
train_df["label_name"] = os.listdir("C:/Users/Karan Chilwal/Desktop/Cropped_label")    #change
for idx, i in enumerate(os.listdir("C:/Users/Karan Chilwal/Desktop/Cropped_input/")):   #change
    train_df["frequency"][idx] = i[2:5]                         #check here for int or str
    train_df["reynolds"][idx] = i[7:14]
    if i[-1] == "f":
        train_df["forw_back"][idx] = 1
    elif i[-1] == "b":
        train_df["forw_back"][idx] = 0
    else:
        continue


import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision
from tqdm import tqdm
import matplotlib as plt



class My_Dataset(Dataset):
    def __init__(self, input_path, label_path, csv_path, train = True):     #check for train/test transform
        self.input_path = input_path
        self.label_path = label_path
        self.csv_file = pd.read_csv(csv_path)
        self.train = train
        
    def transform(self, image, label):
        if random.random() > 0.5:
            image = F.hflip(image)
            label = F.hflip(label)
        return image, label
                    
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, index):
        input_name = self.csv_file.iloc[index, 0]
        label_name = self.csv_file.iloc[index, 1]
        input_img = Image.open(os.path.join(self.input_path, input_name)).convert("RGB")
        label_img = Image.open(os.path.join(self.label_path, label_name)).convert("RGB")
        tabular_data = torch.tensor(float(self.csv_file.iloc[index,2:]))
        
        
        #input_img = F.crop(input_img, top, left, height, width)
        #label_img = F.crop(label_img, top, left, height, width)
        
        if self.train:
            input_img, label_img = self.transform(input_img, label_img)
        
        input_img = F.to_tensor(input_img)
        label_img = F.to_tensor(label_img)
        
        input_img = F.normalize(input_img, mean=(0.5,0.5,0.5), std = (0.5,0.5,0.5))
        label_img = F.normalize(label_img, mean=(0.5,0.5,0.5), std = (0.5,0.5,0.5))
        
        return input_img, label_img ,tabular_data



class Encoder_Single_Block_1(nn.Module):
    def __init__(self, input_channels):
        super(Encoder_Single_Block_1, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels*2, kernel_size=4,stride = 2, padding = 1)
        self.batchnorm2d = nn.BatchNorm2d(input_channels*2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm2d(x)
        return x


class Encoder_Single_Block_2(nn.Module):
    def __init__(self, input_channels):
        super(Encoder_Single_Block_2, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=4,stride = 2, padding = 1)
        self.batchnorm2d = nn.BatchNorm2d(input_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm2d(x)
        return x
    

class Decoder_Single_Block_1(nn.Module):
    def __init__(self, input_channels):
        super(Decoder_Single_Block_1, self).__init__()
        self.convtranspose2d = nn.ConvTranspose2d(input_channels,input_channels//2, kernel_size=4, stride = 2, padding=1)
        self.batchnorm2d = nn.BatchNorm2d(input_channels//2)
        
    def forward(self, x, skip_x):
        x = self.convtranspose2d(x)
        x = self.batchnorm2d(x)
        x = torch.cat([x, skip_x], axis=1)
        return x

class Decoder_Single_Block_2(nn.Module):
    def __init__(self, input_channels):
        super(Decoder_Single_Block_2, self).__init__()
        self.convtranspose2d = nn.ConvTranspose2d(input_channels,input_channels//4, kernel_size=4, stride = 2, padding=1)
        self.batchnorm2d = nn.BatchNorm2d(input_channels//4)
        
    def forward(self, x, skip_x):
        x = self.convtranspose2d(x)
        x = self.batchnorm2d(x)
        x = torch.cat([x, skip_x], axis=1)
        return x


class UNet(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels=64):
        super(UNet, self).__init__()
        self.firstconv = nn.Conv2d(input_channels,hidden_channels,kernel_size=4,stride = 2, padding=1)
        self.firstconvbatchnorm2d = nn.BatchNorm2d(hidden_channels)
        self.contract1 = Encoder_Single_Block_1(hidden_channels)
        self.contract2 = Encoder_Single_Block_1(hidden_channels * 2)
        self.contract3 = Encoder_Single_Block_1(hidden_channels * 4)
        self.contract4 = Encoder_Single_Block_2(hidden_channels * 8)
        self.contract5 = Encoder_Single_Block_2(hidden_channels * 8)
        self.contract6 = Encoder_Single_Block_2(hidden_channels * 8)
        self.contract7 = Encoder_Single_Block_2(hidden_channels * 8)
        self.firstconvtranspose = nn.ConvTranspose2d(hidden_channels*8,hidden_channels*8,kernel_size=4,stride = 2, padding =1)
        self.firstconvtransposebatchnorm2d = nn.BatchNorm2d(hidden_channels*8)
        #cat
        self.expand1 = Decoder_Single_Block_1(hidden_channels*16)
        self.expand2 = Decoder_Single_Block_1(hidden_channels*16)
        self.expand3 = Decoder_Single_Block_1(hidden_channels*16)
        self.expand4 = Decoder_Single_Block_2(hidden_channels*16)
        self.expand5 = Decoder_Single_Block_2(hidden_channels*8)
        self.expand6 = Decoder_Single_Block_2(hidden_channels*4)
        self.lastconv = nn.ConvTranspose2d(hidden_channels*2, output_channels, kernel_size=4, stride=2, padding=1)
        self.linear_layer_1 = nn.Linear(515, 514)
        self.relu = nn.ReLU()
        self.linear_layer_batchnorm_1 = nn.BatchNorm1d(514)
        self.linear_layer_2 = nn.Linear(514,512)
        self.linear_layer_batchnorm_2 = nn.BatchNorm1d(512)
        
        
        
    def forward(self, x, freq):
        x0 = self.firstconv(x) #x0=[n,64,128,128] : x = [n,3,256,256]
        x1 = self.firstconvbatchnorm2d(x0) #x1=n,[64,128,128]
        x2 = self.contract1(x1) #x2=[n,128,64,64]
        x3 = self.contract2(x2) #x3=n,[256,32,32]
        x4 = self.contract3(x3) #x4=[n,512,16,16]
        x5 = self.contract4(x4) #x5=[n,512,8,8]
        x6 = self.contract5(x5) #x6=[n,512,4,4]
        x7 = self.contract6(x6) #x7=[n, 512,2,2]
        x8 = self.contract7(x7) #x8=[n,512,1,1]
        x8_new = x8.view(-1,512*1*1)       #[n,512], freq = [n, 4]
        x8_new = torch.cat([x8_new,freq], axis = 1) #x8_new = [n,515]
        x8_linear_layer_1 = self.linear_layer_1(x8_new)     #[n,514]
        x8_relu_1 = self.relu(x8_linear_layer_1)            #[n,514]
        x8_batchnorm_1 = self.linear_layer_batchnorm_1(x8_relu_1)     #[n,514]
        x8_linear_layer_2 = self.linear_layer_2(x8_batchnorm_1)     # [n,512]
        x8_relu_2 = self.relu(x8_linear_layer_2)        #[n,512]
        x8_batchnorm_2 = self.linear_layer_batchnorm_2(x8_relu_2)       #[n,512]
        x8_to_high_dim = x8_batchnorm_2[:,:,None,None]        #[n,512,1,1]
        x9 = self.firstconvtranspose(x8_to_high_dim) #x9=[n,512,2,2]
        x10 = self.firstconvtransposebatchnorm2d(x9) #x10=[n,512,2,2]
        x11 = torch.cat([x10,x7], axis =1) #x11=[n,1024,2,2]
        x12 = self.expand1(x11,x6) #x12=[n,1024,4,4]
        x13 = self.expand2(x12,x5) #x13=[n,1024,8,8]
        x14 = self.expand3(x13,x4) #x14=[n,1024,16,16]
        x15 = self.expand4(x14,x3) #x15=[n,512,32,32]
        x16 = self.expand5(x15,x2) #x16=[n,256,64,64]
        x17 = self.expand6(x16,x1) #x17=[n,128,128,128]
        x18 = self.lastconv(x17) #x18=[n,3,256,256]'''
        return x18
    
    
lr = 0.0002
device = 'cpu'
unet = UNet(3,3).to(device)
unet_opt = torch.optim.Adam(unet.parameters(), lr=lr)
from pytorch_model_summary import summary
print(summary(unet, torch.zeros((5, 3, 256, 256)), torch.zeros((5,3)),show_input=True,print_summary=False))
print(summary(unet, torch.zeros((5, 3, 256, 256)), torch.zeros((5,3)),show_input=False))

batch_size = 5
input_path = "C:/Users/Karan Chilwal/Desktop/Cropped_input"
label_path = "C:/Users/Karan Chilwal/Desktop/Cropped_label"
csv_path = "C:/Users/Karan Chilwal/Desktop/train_csv.csv"

data = My_Dataset(input_path, label_path, csv_path)              #change this
train_set, test_set = torch.utils.data.random_split(data, [90,30])
len(data)
data[1]

trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
validloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

'''dataiter = iter(trainloader)
images, labels, freq = dataiter.next()
print(type(images))
print(freq.shape)
print(type(freq))
print(images.shape)
print(labels.shape)
print(freq)
freq = freq[:, None]
'''






