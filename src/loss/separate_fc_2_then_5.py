
dir_csv = '../../input/'
dir_train_img = '../../input/stage_1_train_pngs/'
dir_test_img = '../../input/stage_1_test_pngs/'


# Parameters

n_classes = 6
n_epochs = 5
batch_size = 32
threshold = 0.5


import glob
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.optim as optim
from albumentations import Compose, ShiftScaleRotate, CenterCrop, HorizontalFlip, RandomBrightnessContrast
from albumentations.pytorch import ToTensor
from skimage.transform import resize
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm as tqdm

from apex import amp

CT_LEVEL = 40
CT_WIDTH = 150


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
        file_path = os.path.join(self.data_dir, self.data.loc[idx, 'Image'] + '.png')
        from pathlib import Path
        if not Path(file_path).is_file():
            return self.__getitem__(idx + 1)
        # img = self._load_dicom_to_image(file_path)
        img = cv2.imread(file_path)
        if self.transform:       
            augmented = self.transform(image=img)
            img = augmented['image']   
        if self.labels:
            labels = torch.tensor(
                self.data.loc[idx, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])
            return {'image': img, 'labels': (labels[:5], labels[5])}
        
        else:      
            return {'image': img}


class SepalateFc(nn.Module):
    def __init__(self, input_size):
        super(SepalateFc, self).__init__()
        self.fc_5 = torch.nn.Linear(input_size, 5)
        self.fc_2 = torch.nn.Linear(input_size, 1)

    def forward(self, x):
        out_5 = self.fc_5(x)
        out_2 = self.fc_2(x)

        return out_5, out_2


if __name__ == '__main__':
    if not Path('../../src/train.csv').is_file():
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
    # device = torch.device("cpu")
    # model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
    model = torch.hub.load('pytorch/vision', 'shufflenet_v2_x1_0', pretrained=True)
    model.fc = SepalateFc(1024)

    model.to(device)

    crt_5 = torch.nn.BCEWithLogitsLoss()
    crt_2 = torch.nn.BCEWithLogitsLoss()
    plist = [{'params': model.parameters(), 'lr': 2e-5}]
    optimizer = optim.Adam(plist, lr=2e-5)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    for epoch in range(n_epochs):

        print('Epoch {}/{}'.format(epoch, n_epochs - 1))
        print('-' * 10)

        model.train()
        tr_loss = 0

        tk0 = tqdm(data_loader_train, desc="Iteration")

        # 1回目の学習
        for step, batch in enumerate(tk0):

            inputs = batch["image"]
            labels_5, labels_2 = batch["labels"]

            inputs = inputs.to(device, dtype=torch.float)
            # labels_5 = labels_5.to(device, dtype=torch.float)
            labels_2 = labels_2.to(device, dtype=torch.float)

            out_5, out_2 = model(inputs)
            # loss_5 = crt_5(out_5, labels_5)
            loss_2 = crt_2(out_2.reshape((-1, 1)), labels_2.reshape((-1, 1)))

            loss = loss_2

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

        # 2回目の学習
        for epoch in range(n_epochs):

            print('Epoch {}/{}'.format(epoch, n_epochs - 1))
            print('-' * 10)

            model.train()
            tr_loss = 0

            tk0 = tqdm(data_loader_train, desc="Iteration")

            for step, batch in enumerate(tk0):

                inputs = batch["image"]
                labels_5, labels_2 = batch["labels"]

                inputs = inputs.to(device, dtype=torch.float)
                labels_5 = labels_5.to(device, dtype=torch.float)
                # labels_2 = labels_2.to(device, dtype=torch.float)

                out_5, out_2 = model(inputs)
                loss_5 = crt_5(out_5, labels_5)
                # loss_2 = crt_2(out_2.reshape((-1, 1)), labels_2.reshape((-1, 1)))

                loss = loss_5

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

    test_pred = np.zeros((len(test_dataset), n_classes))

    for i, x_batch in enumerate(tqdm(data_loader_test)):

        x_batch = x_batch["image"]
        x_batch = x_batch.to(device, dtype=torch.float)

        with torch.no_grad():

            pred_5, pred_2 = model(x_batch)

            test_pred[i * batch_size:(i + 1) * batch_size, 0:5] = torch.sigmoid(pred_5).detach().cpu()
            test_pred[i * batch_size:(i + 1) * batch_size, 5] = torch.sigmoid(pred_2.reshape((-1,))).detach().cpu()

        # if i > 50:
        #     break

    # Submission

    submission =  pd.read_csv(os.path.join(dir_csv, 'stage_1_sample_submission.csv'))
    submission = pd.concat([submission.drop(columns=['Label']), pd.DataFrame(test_pred.reshape((-1, 1)))], axis=1)
    submission.columns = ['ID', 'Label']
    submission.to_csv(f'../../output/{Path(__file__).name}_sub.csv', index=False)
    submission.head()
