dir_csv = '../input/'
dir_train_img = '../input/stage_1_train_images/'
dir_test_img = '../input/stage_1_test_images/'
dir_train_png = '../input/stage_1_train_pngs/'
dir_test_png = '../input/stage_1_test_pngs/'


n_classes = 6
CT_LEVEL = 40
CT_WIDTH = 150
nn_input_shape = (224, 224)

import glob
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from skimage.transform import resize
from tqdm import tqdm_notebook as tqdm


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


def resize_(image):
    image = resize(image, nn_input_shape)
    return image


def fill_channels(image):
    filled_image = np.stack((image,)*3, axis=-1)
    return filled_image


def _get_hounsfield_window(dicom):
    hu_image = rescale_pixelarray(dicom)
    windowed_image = set_manual_window(hu_image, CT_LEVEL, CT_WIDTH)
    return windowed_image


def _load_dicom_to_image(file_path):
    dicom = pydicom.dcmread(file_path)
    windowed_image = _get_hounsfield_window(dicom)
    image = fill_channels(resize_(windowed_image))
    return image


if __name__ == '__main__':
    train = pd.read_csv(os.path.join(dir_csv, 'stage_1_train.csv'))
    test = pd.read_csv(os.path.join(dir_csv, 'stage_1_sample_submission.csv'))

    train[['ID', 'Image', 'Diagnosis']] = train['ID'].str.split('_', expand=True)
    train = train[['Image', 'Diagnosis', 'Label']]
    train.drop_duplicates(inplace=True)
    train = train.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
    train['Image'] = 'ID_' + train['Image']

    png = glob.glob(os.path.join(dir_train_img, '*.dcm'))
    png = [os.path.basename(png)[:-4] for png in png]
    png = np.array(png)

    train = train[train['Image'].isin(png)]
    train.to_csv('train.csv', index=False)

    test[['ID','Image','Diagnosis']] = test['ID'].str.split('_', expand=True)
    test['Image'] = 'ID_' + test['Image']
    test = test[['Image', 'Label']]
    test.drop_duplicates(inplace=True)

    test.to_csv('test.csv', index=False)

    Path(dir_train_png).mkdir(exist_ok=True)
    Path(dir_test_png).mkdir(exist_ok=True)

    for idx in tqdm(list(train.index)):
        file_path = os.path.join(dir_train_img, train.loc[idx, 'Image'] + '.dcm')
        if not Path(file_path).is_file():
            continue
        img = _load_dicom_to_image(file_path)
        cv2.imwrite(f"{dir_train_png}/{train.loc[idx, 'Image']}.png", img)

    #
    # for idx in tqdm(list(test.index)):
    #     file_path = os.path.join(dir_test_img, test.loc[idx, 'Image'] + '.dcm')
    #     if not Path(file_path).is_file():
    #         continue
    #     img = _load_dicom_to_image(file_path)
    #     cv2.imwrite(f"{dir_test_png}/{test.loc[idx, 'Image']}.png", img)
