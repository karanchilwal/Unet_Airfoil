# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 12:30:56 2021

@author: Chilwal
"""
import pandas as pd
import os
from PIL import Image
import random

train_df = pd.DataFrame(columns = ["input_name", "label_name", "frequency"])
print(train_df)
train_df["input_name"] = os.listdir("C:/Users/Karan Chilwal/Desktop/Cropped_input")    #change
train_df["label_name"] = os.listdir("C:/Users/Karan Chilwal/Desktop/Cropped_label")    #change
for idx, i in enumerate(os.listdir("C:/Users/Karan Chilwal/Desktop/Cropped_input/")):   #change
    train_df["frequency"][idx] = i[2:5]                         #check here for int or str

train_df.to_csv(r'C:/Users/Karan Chilwal/Desktop/train_csv.csv', index = False , header = True)

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
        freq = torch.tensor(float(self.csv_file.iloc[index,2]))
        
        
        #input_img = F.crop(input_img, top, left, height, width)
        #label_img = F.crop(label_img, top, left, height, width)
        
        if self.train:
            input_img, label_img = self.transform(input_img, label_img)
        
        input_img = F.to_tensor(input_img)
        label_img = F.to_tensor(label_img)
        
        input_img = F.normalize(input_img, mean=(0.5,0.5,0.5), std = (0.5,0.5,0.5))
        label_img = F.normalize(label_img, mean=(0.5,0.5,0.5), std = (0.5,0.5,0.5))
        
        return input_img, label_img ,freq


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
        self.linear_layer = nn.Linear(513, 512)
        self.relu = nn.ReLU()
        self.linear_layer_batchnorm = nn.BatchNorm1d(512)
        
        
    def forward(self, x, freq):
        x0 = self.firstconv(x) #x0=[64,128,128] : x = [3,256,256]
        x1 = self.firstconvbatchnorm2d(x0) #x1=[64,128,128]
        x2 = self.contract1(x1) #x2=[128,64,64]
        x3 = self.contract2(x2) #x3=[256,32,32]
        x4 = self.contract3(x3) #x4=[512,16,16]
        x5 = self.contract4(x4) #x5=[512,8,8]
        x6 = self.contract5(x5) #x6=[512,4,4]
        x7 = self.contract6(x6) #x7=[512,2,2]
        x8 = self.contract7(x7) #x8=[512,1,1]
        x8_new = x8.view(-1,512*1*1)       #[512]
        x8_new = torch.cat([x8_new,freq], axis = 1) #x8_new = [513]
        x8_linear_layer = self.linear_layer(x8_new)     #[512,1]
        x8_relu = self.relu(x8_linear_layer)            #[512,1]
        x8_batchnorm = self.linear_layer_batchnorm(x8_relu)     #[512,1]
        x8_to_high_dim = x8_batchnorm[:,:,None,None]        #[512,1,1]
        x9 = self.firstconvtranspose(x8_to_high_dim) #x9=[512,2,2]
        x10 = self.firstconvtransposebatchnorm2d(x9) #x10=[512,2,2]
        x11 = torch.cat([x10,x7], axis =1) #x11=[1024,2,2]
        x12 = self.expand1(x11,x6) #x12=[1024,4,4]
        x13 = self.expand2(x12,x5) #x13=[1024,8,8]
        x14 = self.expand3(x13,x4) #x14=[1024,16,16]
        x15 = self.expand4(x14,x3) #x15=[512,32,32]
        x16 = self.expand5(x15,x2) #x16=[256,64,64]
        x17 = self.expand6(x16,x1) #x17=[128,128,128]
        x18 = self.lastconv(x17) #x18=[3,256,256]'''
        return x18
    
           

criterion = nn.MSELoss()
n_epochs = 100 
lr = 0.0002
device = 'cpu'
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
unet = UNet(3,3).to(device)
unet_opt = torch.optim.Adam(unet.parameters(), lr=lr)
from pytorch_model_summary import summary
print(summary(unet, torch.zeros((5, 3, 256, 256)), torch.zeros((5,1)),show_input=True,print_summary=False))
print(summary(unet, torch.zeros((5, 3, 256, 256)), torch.zeros((5,1)),show_input=False))


loop_step = 0
print_step = 10


def show_tensor_images(image_tensor, num_images=5, size=(3, 256, 256)):
    image_tensor = image_tensor/2 +0.5
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = torchvision.utils.make_grid(image_unflat[:num_images], nrow=5)
    plt.pyplot.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.pyplot.show()
    
train_loss = []
valid_loss = []    

for epoch in range(n_epochs):
    print("Epoch:", epoch+1)
    inner_loop_step = 0
    running_loss = 0.0
    running_loss_valid = 0.0
    
    unet.train()
    for input_img,label_img,freq in tqdm(trainloader):
        #image_width = image.shape[3]
        #input_image = image[:, :, :, :image_width // 2]
        #input_image = nn.functional.interpolate(input_image, size= 256)     #[n,3,256,256]
        #label = image[:, :, :, image_width // 2:]
        #label = nn.functional.interpolate(label, size=256)              #[n,3,256,256]
        freq = freq[:,None]
        
        input_img = input_img.to(device)
        label_img = label_img.to(device)
        freq = freq.to(device)
        
        
        unet_opt.zero_grad()
        pred = unet(input_img,freq)
        unet_loss = criterion(pred, label_img)
        unet_loss.backward()
        unet_opt.step()
        
        running_loss += unet_loss.item()
        
        #if loop_step % print_step == 0:
        #print("Epoch: {} , Steps inside Each Epoch: {}".format(epoch+1, inner_loop_step+1))
        inner_loop_step += 1
        loop_step += 1
    
    train_loss.append(running_loss/len(trainloader))
    
    unet.eval()
    for input_valid, label_valid, freq_valid in tqdm(validloader):
        #image_valid_width = image_valid.shape[3]
        #input_valid = image_valid[:,:,:,:image_valid_width//2]
        #input_valid = nn.functional.interpolate(input_valid, size = 256)
        #label_valid = image_valid[:,:,:,image_valid_width//2:]
        #label_valid =  nn.functional.interpolate(label_valid, size = 256)
        freq_valid = freq_valid[:, None]
        input_valid = input_valid.to(device)
        label_valid = label_valid.to(device)
        freq_valid = freq_valid.to(device)
        
        pred_valid = unet(input_valid, freq_valid)
        unet_loss_valid = criterion(pred_valid, label_valid)
        
        running_loss_valid += unet_loss_valid.item()
        
    valid_loss.append(running_loss_valid/len(validloader))
    
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch+1, running_loss/len(trainloader), running_loss_valid/len(validloader)))
    plt.pyplot.plot(train_loss, label = "Training loss")
    plt.pyplot.plot(valid_loss, label = "Validation Loss")
    plt.pyplot.title("Unet Losses")
    plt.pyplot.legend()
    plt.pyplot.draw()
    plt.pyplot.show()
        
        
        
    if epoch % print_step == 0:
        
        show_tensor_images(input_img)
        show_tensor_images(label_img)
        show_tensor_images(pred)
        show_tensor_images(input_valid)
        show_tensor_images(label_valid)
        show_tensor_images(pred_valid)
        
        
        
#printing weight and bias matrix        
for param_tensor in unet.state_dict():
    print(param_tensor)
    print(unet.state_dict()[param_tensor])
    print(param_tensor, "\t", unet.state_dict()[param_tensor].size())
    break
#    
plt.pyplot.plot(train_loss[30:], label = "Training loss") 
plt.pyplot.plot(valid_loss[30:], label = "Validation Loss")
plt.pyplot.title("Unet Losses")
plt.pyplot.legend()
plt.pyplot.draw()
plt.pyplot.show()

for param_tensor in unet.state_dict():
    print(param_tensor, "\t", unet.state_dict()[param_tensor].size())
    
#torch.save(unet.state_dict(), "K:/Deep Learning/data and processing/results MSELoss/saved models")


#torch.cuda.is_available()