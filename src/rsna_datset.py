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
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt
from torchvision import transforms


class IntracranialDataset(Dataset):

    def __init__(self, csv_file, path, labels, transform=None):

        self.path = path
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = labels

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        img_name = os.path.join(self.path, self.data.loc[idx, 'Image'] + '.png')
        img = cv2.imread(img_name)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        if self.labels:

            labels = torch.tensor(
                self.data.loc[
                    idx, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])
            return {'image': img, 'labels': labels}

        else:

            return {'image': img}