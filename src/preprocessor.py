# code is from https://www.kaggle.com/allunia/rsna-ih-detection-baseline

import numpy as np
import pydicom

from skimage.transform import resize
from imgaug import augmenters as iaa


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


class Preprocessor:

    def __init__(self, path, backbone, ct_level, ct_width, augment=False):
        self.path = path
        self.backbone = backbone
        self.nn_input_shape = backbone["nn_input_shape"]
        self.ct_level = ct_level
        self.ct_width = ct_width
        self.augment = augment

    # 1. We need to load the dicom dataset
    def load_dicom_dataset(self, filename):
        dataset = pydicom.dcmread(self.path + filename)
        return dataset

    # 2. We need to rescale the pixelarray to Hounsfield units
    #    and we need to focus on our custom window:
    def get_hounsfield_window(self, dataset, level, width):
        hu_image = rescale_pixelarray(dataset)
        windowed_image = set_manual_window(hu_image, level, width)
        return windowed_image

    # 3. Resize the image to the input shape of our CNN
    def resize(self, image):
        image = resize(image, self.nn_input_shape)
        return image

    # 4. If we like to augment our image, let's do it:
    def augment_img(self, image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                # iaa.Affine(rotate=(-4, 4)),
                iaa.Fliplr(0.5),
            ])], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug

    def fill_channels(self, image):
        filled_image = np.stack((image,) * 3, axis=-1)
        return filled_image

    def preprocess(self, identifier):
        filename = identifier + ".dcm"
        dataset = self.load_dicom_dataset(filename)
        windowed_image = self.get_hounsfield_window(dataset, self.ct_level, self.ct_width)
        image = self.resize(windowed_image)
        if self.augment:
            image = self.augment_img(image)
        image = self.fill_channels(image)
        return image

    def normalize(self, image):
        image = 2 * (image / 255) - 1
        return image


