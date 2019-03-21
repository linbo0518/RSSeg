import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils import data

mean_value = (0.5, 0.5, 0.5)
std_value = (0.23)


class RSClsDataset(data.Dataset):

    def __init__(self, csv_file, root_dir, usage):
        assert usage == 'train' or usage == 'valid'
        self._df = pd.read_csv(csv_file)
        self._root_dir = root_dir
        if usage == 'train':
            self._transformer = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=mean_value, std=std_value, p=1.0)
            ])
        else:
            self._transformer = A.Normalize(
                mean=mean_value, std=std_value, p=1.0)

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self._root_dir, self._df.iloc[idx, 0]))
        label = np.array([self._df.iloc[idx, 1], 1 - self._df.iloc[idx, 1]],
                         dtype='float32')

        image = self._transformer(image=image)['image']
        image = np.transpose(image, (2, 0, 1))

        return image, label


class RSSegDataset(data.Dataset):

    def __init__(self, csv_file, image_root_dir, mask_root_dir, usage):
        assert usage == 'train' or usage == 'valid'
        self._df = pd.read_csv(csv_file)
        self._image_root_dir = image_root_dir
        self._mask_root_dir = mask_root_dir
        self._usage = usage
        self._normalizer = A.Normalize(mean=mean_value, std=std_value, p=1.0)
        if self._usage == 'train':
            self._transformer = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5)
            ])

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        image = cv2.imread(
            os.path.join(self._image_root_dir, self._df.iloc[idx, 0]))
        mask = cv2.imread(
            os.path.join(self._mask_root_dir, self._df.iloc[idx, 0]),
            cv2.IMREAD_GRAYSCALE)
        if self._usage == 'train':
            augmented = self._transformer(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        image = self._normalizer(image=image)['image']
        image = np.transpose(image, (2, 0, 1))
        mask = mask / 255
        mask = mask.astype('float32')
        mask = np.expand_dims(mask, 0)
        return image, mask


class RSMTDataset(data.Dataset):

    def __init__(self, csv_file, image_root_dir, mask_root_dir, usage):
        assert usage == 'train' or usage == 'valid'
        self._df = pd.read_csv(csv_file)
        self._image_root_dir = image_root_dir
        self._mask_root_dir = mask_root_dir
        self._usage = usage
        self._normalizer = A.Normalize(mean=mean_value, std=std_value, p=1)
        if self._usage == 'train':
            self._transformer = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5)
            ])

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        image = cv2.imread(
            os.path.join(self._image_root_dir, self._df.iloc[idx, 0]))
        mask = cv2.imread(
            os.path.join(self._mask_root_dir, self._df.iloc[idx, 0]),
            cv2.IMREAD_GRAYSCALE)
        label = np.array([self._df.iloc[idx, 1], 1 - self._df.iloc[idx, 1]],
                         dtype='float32')
        if self._usage == 'train':
            augmented = self._transformer(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        image = self._normalizer(image=image)['image']
        image = np.transpose(image, (2, 0, 1))
        mask = mask.astype('float32')
        mask /= 255
        mask = np.expand_dims(mask, 0)
        return image, label, mask