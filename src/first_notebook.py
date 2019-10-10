#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Using mixed precision along with efficientnet-b0 and a little bit of pre-processing, a single pass of the entire 670k image dataset should take approx. 45m (at 224x224 resolution).

# # Sources
# 
# Windowing functions for pre-processed data taken from the following:
# 
# - https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing 

# # Parameters

# In[1]:


# Input

dir_csv = '../input/'
dir_train_img = '../input/stage_1_train_images/'
dir_test_img = '../input/stage_1_test_images/'


# In[2]:



# Parameters

n_classes = 6
n_epochs = 5
batch_size = 8


# # Setup
# 
# Need to grab a couple of extra libraries
# 
# - Nvidia Apex for mixed precision training (https://github.com/NVIDIA/apex)
# - Pytorch implementation of efficientnet (https://github.com/lukemelas/EfficientNet-PyTorch)

# In[3]:


# Installing useful libraries

#get_ipython().system('git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./')
# !cd ../apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./    
#get_ipython().system('pip install --upgrade efficientnet-pytorch')
    


# In[4]:


# Libraries

# from apex import amp
import os
import cv2
import glob
from skimage.transform import resize
import pydicom
import numpy as np
import pandas as pd
from efficientnet_pytorch import EfficientNet
import torch
import torch.optim as optim
from albumentations import Compose, ShiftScaleRotate, Resize
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt


# In[5]:


CT_LEVEL = 40
CT_WIDTH = 150

LR = 0.001


# In[6]:


def rescale_pixelarray(dataset):
    image = dataset.pixel_array
    rescaled_image = image * dataset.RescaleSlope + dataset.RescaleIntercept
    rescaled_image[rescaled_image < -1024] = -1024
    return rescaled_image


def set_manual_window(hu_image, custom_center, custom_width):
    min_value = custom_center - (custom_width / 2)
    max_value = custom_center + (custom_width / 2)
    hu_image[hu_image < min_value] = min_value
    hu_image[hu_image > max_value] = max_value
    return hu_image


# In[7]:



# Functions

class IntracranialDataset(Dataset):

    def __init__(self, csv_file, data_dir, labels, ct_level=CT_LEVEL, ct_width=CT_WIDTH, transform=None):
        
        self.data_dir = data_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = labels
        self.level = ct_level
        self.width = ct_width
        self.nn_input_shape = (224, 224)

    def __len__(self):
        return len(self.data)
        
    def resize(self, image):
        image = resize(image, self.nn_input_shape)
        return image
    
    def fill_channels(self, image):
        filled_image = np.stack((image,)*3, axis=-1)
        return filled_image
    
    def _get_hounsfield_window(self, dicom):
        hu_image = rescale_pixelarray(dicom)
        windowed_image = set_manual_window(hu_image, self.level, self.width)
        return windowed_image
    
    def _load_dicom_to_image(self, file_path):
        dicom = pydicom.dcmread(file_path)
        windowed_image = self._get_hounsfield_window(dicom)
        image = self.fill_channels(self.resize(windowed_image))
        return image

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.data.loc[idx, 'Image'] + '.dcm')
        from pathlib import Path
        if not Path(file_path).is_file():
            return self.__getitem__(idx + 1)
        img = self._load_dicom_to_image(file_path)
        if self.transform:       
            augmented = self.transform(image=img)
            img = augmented['image']   
        if self.labels:
            labels = torch.tensor(
                self.data.loc[idx, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])
            return {'image': img, 'labels': labels}    
        
        else:      
            return {'image': img}


# # CSV

# In[8]:


# CSVs

train = pd.read_csv(os.path.join(dir_csv, 'stage_1_train.csv'))
test = pd.read_csv(os.path.join(dir_csv, 'stage_1_sample_submission.csv'))


# In[9]:



# Split train out into row per image and save a sample

train[['ID', 'Image', 'Diagnosis']] = train['ID'].str.split('_', expand=True)
train = train[['Image', 'Diagnosis', 'Label']]
train.drop_duplicates(inplace=True)
train = train.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
train['Image'] = 'ID_' + train['Image']
train.head()


# In[10]:


# Some files didn't contain legitimate images, so we need to remove them

png = glob.glob(os.path.join(dir_train_img, '*.dcm'))
png = [os.path.basename(png)[:-4] for png in png]
png = np.array(png)

train = train[train['Image'].isin(png)]
train.to_csv('train.csv', index=False)


# In[11]:


# Also prepare the test data

test[['ID','Image','Diagnosis']] = test['ID'].str.split('_', expand=True)
test['Image'] = 'ID_' + test['Image']
test = test[['Image', 'Label']]
test.drop_duplicates(inplace=True)

test.to_csv('test.csv', index=False)


# # DataLoaders

# In[12]:


# Data loaders

transform_train = Compose([
    ShiftScaleRotate(),
    ToTensor()
])

transform_test= Compose([
    ToTensor()
])

train_dataset = IntracranialDataset(
    csv_file='train.csv', data_dir=dir_train_img, transform=transform_train, labels=True)

test_dataset = IntracranialDataset(
    csv_file='test.csv', data_dir=dir_test_img, transform=transform_test, labels=False)

data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)


# In[13]:


# # Model

# In[15]:


# Model

device = torch.device("cuda:0")
model = EfficientNet.from_pretrained('efficientnet-b0') 
model._fc = torch.nn.Linear(1280, n_classes)

model.to(device)

criterion = torch.nn.BCEWithLogitsLoss()
plist = [{'params': model.parameters(), 'lr': 2e-5}]
optimizer = optim.Adam(plist, lr=2e-5)

# model, optimizer = amp.initialize(model, optimizer, opt_level="O1")


# # Training

# In[16]:


# Train


for epoch in range(n_epochs):
    
    print('Epoch {}/{}'.format(epoch, n_epochs - 1))
    print('-' * 10)

    model.train()    
    tr_loss = 0
    
    tk0 = tqdm(data_loader_train, desc="Iteration")

    for step, batch in enumerate(tk0):

        inputs = batch["image"]
        labels = batch["labels"]

        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

#         with amp.scale_loss(loss, optimizer) as scaled_loss:
#             scaled_loss.backward()
        loss.backward()

        tr_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()

    epoch_loss = tr_loss / len(data_loader_train)
    print('Training Loss: {:.4f}'.format(epoch_loss))


# # Inference

# In[17]:


# Inference

for param in model.parameters():
    param.requires_grad = False

model.eval()

test_pred = np.zeros((len(test_dataset) * n_classes, 1))
from tqdm import tqdm

for i, x_batch in tqdm(enumerate(tqdm(data_loader_test))):
    
    x_batch = x_batch["image"]
    x_batch = x_batch.to(device, dtype=torch.float)
    
    with torch.no_grad():
        
        pred = model(x_batch)
        
        test_pred[(i * batch_size * n_classes):((i + 1) * batch_size * n_classes)] = torch.sigmoid(
            pred).detach().cpu().reshape((len(x_batch) * n_classes, 1))


# # Submission

# In[ ]:


# Submission

submission =  pd.read_csv(os.path.join(dir_csv, 'stage_1_sample_submission.csv'))
submission = pd.concat([submission.drop(columns=['Label']), pd.DataFrame(test_pred)], axis=1)
submission.columns = ['ID', 'Label']

submission.to_csv('submission.csv', index=False)
submission.head()


# # Clean Up
# 
# Have to clean up since Kaggle limits the number of files that can be output from a kernel

# In[ ]:


get_ipython().system('rm -rf /kaggle/working/apex')
get_ipython().system('rm test.csv')
get_ipython().system('rm train.csv')


# In[ ]:




