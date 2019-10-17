import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from joblib import Parallel, delayed
from skimage.transform import resize
from tqdm import tqdm

TRAIN_IMG_PATH = "../../input/stage_1_train_images/"
TEST_IMG_PATH = "../../input/stage_1_test_images/"
# BASE_PATH = '/kaggle/input/rsna-intracranial-hemorrhage-detection/'
# TRAIN_DIR = 'stage_1_train_images/'
IMAGE_SIZE = (224, 224)
NORMALIZE = True

hem_types = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']


def view_images(images):
    width = 1
    height = 1
    fig, axs = plt.subplots(height, width, figsize=(5, 5))

    for im in range(0, 1):
        image = images[im]
        i = im // width
        j = im % width
        axs.imshow(image, cmap=plt.cm.bone)
        axs.axis('off')
        title = hem_types[im] if im < len(hem_types) else 'normal'
        axs.set_title(title)

    plt.show()


def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


def brain_window(img):
    window_min = 0
    window_max = 80
    _, _, intercept, slope = get_windowing(img)
    img = img.pixel_array
    img = img * slope + intercept
    img[img < window_min] = window_min
    img[img > window_max] = window_max
    if NORMALIZE:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img


def metadata_window(img, print_ranges=False):
    # Get data from dcm
    window_center, window_width, intercept, slope = get_windowing(img)
    img = img.pixel_array

    # Window based on dcm metadata
    img = img * slope + intercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    if print_ranges:
        print(img_min, img_max)
    img[img < img_min] = img_min
    img[img > img_max] = img_max

    # Normalize
    if NORMALIZE:
        img = (img - img_min) / (img_max - img_min)
    return img


def all_channels_window(img):
    grey_img = brain_window(img) * 3.0
    all_chan_img = np.zeros((grey_img.shape[0], grey_img.shape[1], 3))
    all_chan_img[:, :, 2] = np.clip(grey_img, 0.0, 1.0)
    all_chan_img[:, :, 0] = np.clip(grey_img - 1.0, 0.0, 1.0)
    all_chan_img[:, :, 1] = np.clip(grey_img - 2.0, 0.0, 1.0)
    return all_chan_img


def map_to_gradient(grey_img):
    rainbow_img = np.zeros((grey_img.shape[0], grey_img.shape[1], 3))
    rainbow_img[:, :, 0] = np.clip(4 * grey_img - 2, 0, 1.0) * (grey_img > 0) * (grey_img <= 1.0)
    rainbow_img[:, :, 1] = np.clip(4 * grey_img * (grey_img <=0.75), 0,1) + np.clip((-4*grey_img + 4) * (grey_img > 0.75), 0, 1)
    rainbow_img[:, :, 2] = np.clip(-4 * grey_img + 2, 0, 1.0) * (grey_img > 0) * (grey_img <= 1.0)
    return rainbow_img


def rainbow_window(img):
    grey_img = brain_window(img)
    return map_to_gradient(grey_img)


def window_image(img, window_center, window_width):
    _, _, intercept, slope = get_windowing(img)
    img = img.pixel_array * slope + intercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    if NORMALIZE:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img


def bsb_window(img):
    brain_img = window_image(img, 40, 80)
    subdural_img = window_image(img, 80, 200)
    bone_img = window_image(img, 600, 2000)

    bsb_img = np.zeros((brain_img.shape[0], brain_img.shape[1], 3))
    bsb_img[:, :, 0] = brain_img
    bsb_img[:, :, 1] = subdural_img
    bsb_img[:, :, 2] = bone_img
    return bsb_img


def window_image_bottom(img, window_center, window_width):
    _, _, intercept, slope = get_windowing(img)
    img = img.pixel_array * slope + intercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_min
    if NORMALIZE:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img


def exclusive_window(img):
    brain_img = window_image_bottom(img, 40, 80)
    subdural_img = window_image_bottom(img, 80, 200)
    bone_img = window_image_bottom(img, 600, 2000)

    bsb_img = np.zeros((brain_img.shape[0], brain_img.shape[1], 3))
    bsb_img[:, :, 0] = brain_img
    bsb_img[:, :, 1] = subdural_img
    bsb_img[:, :, 2] = bone_img
    return bsb_img


def rainbow_bsb_window(img):
    brain_img = window_image(img, 40, 80)
    subdural_img = window_image(img, 80, 200)
    bone_img = window_image(img, 600, 2000)
    combo = (brain_img*0.3 + subdural_img*0.5 + bone_img*0.2)
    return map_to_gradient(combo)


def sigmoid_window(img, window_center, window_width, U=1.0, eps=(1.0 / 255.0)):
    _, _, intercept, slope = get_windowing(img)
    img = img.pixel_array * slope + intercept
    ue = np.log((U / eps) - 1.0)
    W = (2 / window_width) * ue
    b = ((-2 * window_center) / window_width) * ue
    z = W * img + b
    img = U / (1 + np.power(np.e, -1.0 * z))
    if NORMALIZE:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img


def sigmoid_brain_window(img):
    return sigmoid_window(img, 40, 80)


def sigmoid_bsb_window(img):
    brain_img = sigmoid_window(img, 40, 80)
    subdural_img = sigmoid_window(img, 80, 200)
    bone_img = sigmoid_window(img, 600, 2000)

    bsb_img = np.zeros((brain_img.shape[0], brain_img.shape[1], 3))
    bsb_img[:, :, 0] = brain_img
    bsb_img[:, :, 1] = subdural_img
    bsb_img[:, :, 2] = bone_img
    return bsb_img


def map_to_gradient_sig(grey_img):
    rainbow_img = np.zeros((grey_img.shape[0], grey_img.shape[1], 3))
    rainbow_img[:, :, 0] = np.clip(4*grey_img - 2, 0, 1.0) * (grey_img > 0.01) * (grey_img <= 1.0)
    rainbow_img[:, :, 1] =  np.clip(4*grey_img * (grey_img <=0.75), 0,1) + np.clip((-4*grey_img + 4) * (grey_img > 0.75), 0, 1)
    rainbow_img[:, :, 2] = np.clip(-4*grey_img + 2, 0, 1.0) * (grey_img > 0.01) * (grey_img <= 1.0)
    return rainbow_img


def sigmoid_rainbow_bsb_window(img):
    brain_img = sigmoid_window(img, 40, 80)
    subdural_img = sigmoid_window(img, 80, 200)
    bone_img = sigmoid_window(img, 600, 2000)
    combo = (brain_img*0.35 + subdural_img*0.5 + bone_img*0.15)
    if NORMALIZE:
        combo = (combo - np.min(combo)) / (np.max(combo) - np.min(combo))
    return map_to_gradient_sig(combo)


if __name__ == '__main__':
    WINDOWS = {'brain': brain_window, 'metadata': metadata_window, 'all-channel': all_channels_window,
               'gradient': rainbow_window, 'brain-subdural-bone': bsb_window, 'exclusive': exclusive_window,
               'gradient-b-s-b': rainbow_bsb_window, 'sigmoid': sigmoid_brain_window,
               'sigmoid-b-s-b': sigmoid_bsb_window, 'sigmoid-gradient-b-s-b': sigmoid_rainbow_bsb_window}
    train = pd.read_csv("../../input/stage_1_train.csv")
    train_images = os.listdir(f"../../input/stage_1_train_images/")

    train['filename'] = train['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".dcm")
    train['type'] = train['ID'].apply(lambda st: st.split('_')[2])
    train = train[['Label', 'filename', 'type']].drop_duplicates().pivot(index='filename', columns='type',
                                                                         values='Label').reset_index()

    test = pd.read_csv('../../input/stage_1_sample_submission.csv')
    test[['ID', 'Image', 'Diagnosis']] = test['ID'].str.split('_', expand=True)
    test['Image'] = 'ID_' + test['Image']
    test['filename'] = test['Image'] + '.dcm'
    # test = test[['Image', 'Label']]
    test.drop_duplicates(inplace=True)

    # test.to_csv('test.csv', index=False)

    def windowing(path):
        if not Path(os.path.join(TRAIN_IMG_PATH, path)).is_file():
            return
        dcm = pydicom.dcmread(os.path.join(TRAIN_IMG_PATH, path))
        try:
            img = window_func(dcm) * 255
        except ValueError as e:
            return
        img = resize(img, IMAGE_SIZE)
        cv2.imwrite(os.path.join(window_path, path)[:-4] + '.png', img)

    for window_name, window_func in WINDOWS.items():
        # window_name = 'no'
        window_path = f"../../input/processed/train_{window_name}_{IMAGE_SIZE[0]}"
        Path(window_path).mkdir(exist_ok=True)

        Parallel(n_jobs=-1)([delayed(windowing)(filename) for filename in tqdm(train['filename'])])
        Parallel(n_jobs=-1)([delayed(windowing)(filename) for filename in tqdm(test['filename'])])

        # for filename in tqdm(test['filename']):
        #     window_path = f"../../input/processed/test_{window_name}_{IMAGE_SIZE[0]}"
        #     Path(window_path).mkdir(exist_ok=True)
        #
        #     if not Path(os.path.join(TEST_IMG_PATH, filename)).is_file():
        #         continue
        #
        #     dcm = pydicom.read_file(os.path.join(TEST_IMG_PATH, filename))
        #     img = window_func(dcm) * 255
        #     img = resize(img, IMAGE_SIZE)
        #     cv2.imwrite(os.path.join(window_path, filename)[:-4] + '.png', img)

            # break