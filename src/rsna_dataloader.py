from apex import amp
import os
import cv2
import glob
import pydicom
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from albumentations import Compose, ShiftScaleRotate, Resize, CenterCrop, HorizontalFlip, RandomBrightnessContrast
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
from .rsna_datset import IntracranialDataset
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt
from torchvision import transforms


def set_dataloader():
    transform_train = Compose([CenterCrop(200, 200),
                               #Resize(224, 224),
                               HorizontalFlip(),
                               RandomBrightnessContrast(),
                               ShiftScaleRotate(),
                               ToTensor()
                               ])

    transform_test = Compose([CenterCrop(200, 200),
                             #Resize(224, 224),
                              ToTensor()
                              ])

    train_dataset = IntracranialDataset(
        csv_file='train.csv', path=dir_train_img, transform=transform_train, labels=True)

    test_dataset = IntracranialDataset(
        csv_file='test.csv', path=dir_test_img, transform=transform_test, labels=False)

    data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return data_loader_train, data_loader_test