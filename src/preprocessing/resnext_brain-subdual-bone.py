
dir_csv = '../../input/'
dir_train_img = '../..//input/processed/train_brain-subdural-bone_224/'
dir_test_img = '../input/processed/test_brain-subdural-bone_224/'


# Parameters

n_classes = 6
n_epochs = 5
batch_size = 32


import glob
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from albumentations import Compose, ShiftScaleRotate, CenterCrop, HorizontalFlip, RandomBrightnessContrast
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
from tqdm import tqdm as tqdm

from apex import amp


class IntracranialDataset(Dataset):

    def __init__(self, csv_file, data_dir, labels, ct_level=0, ct_width=0, transform=None):
        
        self.data_dir = data_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = labels
        self.level = ct_level
        self.width = ct_width
        self.nn_input_shape = (224, 224)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.data.loc[idx, 'Image'] + '.png')
        from pathlib import Path
        if not Path(file_path).is_file():
            return self.__getitem__(idx + 1)

        img = cv2.imread(file_path)
        if self.transform:       
            augmented = self.transform(image=img)
            img = augmented['image']   
        if self.labels:
            labels = torch.tensor(
                self.data.loc[idx, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])
            return {'image': img, 'labels': labels}    
        
        else:      
            return {'image': img}


if __name__ == '__main__':
    if not Path('train.csv').is_file():
        train = pd.read_csv(os.path.join(dir_csv, 'stage_1_train.csv'))
        test = pd.read_csv(os.path.join(dir_csv, 'stage_1_sample_submission.csv'))

        # Split train out into row per image and save a sample

        train[['ID', 'Image', 'Diagnosis']] = train['ID'].str.split('_', expand=True)
        train = train[['Image', 'Diagnosis', 'Label']]
        train.drop_duplicates(inplace=True)
        train = train.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
        train['Image'] = 'ID_' + train['Image']
        train.head()

        # Some files didn't contain legitimate images, so we need to remove them

        png = glob.glob(os.path.join(dir_train_img, '*.png'))
        png = [os.path.basename(png)[:-4] for png in png]
        png = np.array(png)

        train = train[train['Image'].isin(png)]
        train.to_csv('train.csv', index=False)

        # Also prepare the test data

        test[['ID','Image','Diagnosis']] = test['ID'].str.split('_', expand=True)
        test['Image'] = 'ID_' + test['Image']
        test = test[['Image', 'Label']]
        test.drop_duplicates(inplace=True)

        test.to_csv('test.csv', index=False)

    # Data loaders

    transform_train = Compose([CenterCrop(200, 200),
                               #Resize(224, 224),
                               HorizontalFlip(),
                               RandomBrightnessContrast(),
        ShiftScaleRotate(),
        ToTensor()
    ])

    transform_test= Compose([CenterCrop(200, 200),
                             #Resize(224, 224),
        ToTensor()
    ])

    train_dataset = IntracranialDataset(
        csv_file='train.csv', data_dir=dir_train_img, transform=transform_train, labels=True)

    test_dataset = IntracranialDataset(
        csv_file='test.csv', data_dir=dir_test_img, transform=transform_test, labels=False)

    data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    device = torch.device("cuda:0")
    model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
    # model = torch.hub.load('pytorch/vision', 'shufflenet_v2_x1_0', pretrained=True)
    model.fc = torch.nn.Linear(1024, n_classes)

    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    plist = [{'params': model.parameters(), 'lr': 2e-5}]
    optimizer = optim.Adam(plist, lr=2e-5)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

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

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            # loss.backward()

            tr_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()

            if epoch == 1 and step > 6000:
                epoch_loss = tr_loss / 6000
                print('Training Loss: {:.4f}'.format(epoch_loss))
                break

        epoch_loss = tr_loss / len(data_loader_train)
        print('Training Loss: {:.4f}'.format(epoch_loss))

    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    test_pred = np.zeros((len(test_dataset) * n_classes, 1))

    for i, x_batch in enumerate(tqdm(data_loader_test)):

        x_batch = x_batch["image"]
        x_batch = x_batch.to(device, dtype=torch.float)

        with torch.no_grad():

            pred = model(x_batch)

            test_pred[(i * batch_size * n_classes):((i + 1) * batch_size * n_classes)] = torch.sigmoid(
                pred).detach().cpu().reshape((len(x_batch) * n_classes, 1))

    # Submission

    submission =  pd.read_csv(os.path.join(dir_csv, 'stage_1_sample_submission.csv'))
    submission = pd.concat([submission.drop(columns=['Label']), pd.DataFrame(test_pred)], axis=1)
    submission.columns = ['ID', 'Label']
    submission.to_csv(f'../../output/{Path(__file__).name}_sub.csv', index=False)
    submission.head()
