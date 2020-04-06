import torch
import numpy as np
import pandas as pd
import joblib
import albumentations
import os
import glob
import cv2
from albumentations.core.transforms_interface import ImageOnlyTransform

def cutoutside(img,bin_width, fill_value):
    # Make a copy of the input image since we don't want to modify it directly
    img = img.copy()
    h, w = img.shape[:2]
    img[:,0:bin_width,:] = fill_value
    img[:,h-bin_width:h,:] = fill_value
    img[0:bin_width,:,:] = fill_value
    img[w-bin_width:w,:,:] = fill_value
    return img

class OutsideCutout(ImageOnlyTransform):
    def __init__(self, bin_size=30, always_apply=False, fill_value=0, p=0.5):
        super(OutsideCutout, self).__init__(always_apply, p)
        self.bin_size= bin_size
        self.fill_values = fill_value
    def apply(self, image, **params):
        return cutoutside(image,bin_width=self.bin_size, fill_value=self.fill_values)

class ImageDataset:
    def __init__(self,fold_file, pkl_file_path, folds, image_height, image_width, mean, std):
        self.pkl_file_path = pkl_file_path
        self.fold_file = fold_file

        df = pd.read_csv(self.fold_file)
        df = df[['image_id','labels','kfold']]
        df = df[df['kfold'].isin(folds)].reset_index(drop= True)

        class_map = {'A':0,'B':1,'C':2}

        self.img_id = df['image_id'].apply(lambda x: x.split('.')[0]).values
        self.labels = df['labels'].apply(lambda x: x[-1]).map(class_map).values

        if len(folds)==1:
            # validation set
            self.aug = albumentations.Compose([
                albumentations.Resize(image_height,image_width,always_apply=True),
                albumentations.Normalize(mean,std,always_apply=True),

            ])
        else:
            # training set
            self.aug = albumentations.Compose([
                albumentations.Resize(image_height, image_width, always_apply=True),
                albumentations.ShiftScaleRotate(shift_limit=0.0625,
                                                scale_limit=0.1,
                                                rotate_limit=5,
                                                p=0.9),
                albumentations.Normalize(mean, std, always_apply=True)
            ])

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, item):
        image = joblib.load(f"{self.pkl_file_path}/{self.img_id[item]}.pkl")
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image, [2,0,1]).astype(float) # for using torchvision model
        return {
            'image':torch.tensor(image, dtype=torch.float),
            'label': torch.tensor(self.labels[item], dtype=torch.long)
        }


class ImageTestDataset:
    def __init__(self, file_path, image_height, image_width, mean, std):
        self.image_files = glob.glob(os.path.join(file_path, '*.jpg'))
        self.image_ids = [os.path.basename(f).split('.')[0] for f in self.image_files]

        # validation set
        self.aug = albumentations.Compose([
            albumentations.Resize(image_height,image_width,always_apply=True),
            albumentations.Normalize(mean,std,always_apply=True),
        ])
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        img_bgr = cv2.imread(self.image_files[item])
        img_rgb = img_bgr[:, :, [2, 1, 0]]
        img = self.aug(image=np.array(img_rgb))['image']
        img_float = np.transpose(img, [2, 0, 1]).astype(float) # for using torchvision model
        return {
            'image' : torch.tensor(img_float, dtype=torch.float),
            'image_id' : self.image_ids[item]
        }

class ImageExpDataset:
    def __init__(self,fold_file, pkl_file_path, folds, image_height, image_width, mean, std):
        self.pkl_file_path = pkl_file_path
        self.fold_file = fold_file

        df = pd.read_csv(self.fold_file)
        df = df[['image_id','labels','kfold']]
        df = df[df['kfold'].isin(folds)].reset_index(drop= True)

        class_map = {'A':0,'B':1,'C':2}

        self.img_id = df['image_id'].apply(lambda x: x.split('.')[0]).values
        self.labels = df['labels'].apply(lambda x: x[-1]).map(class_map).values


        # training set
        self.aug = albumentations.Compose([
            albumentations.Resize(image_height, image_width, always_apply=True),
            albumentations.RandomShadow(shadow_roi=(0, 0.85, 1, 1), p=0.5),
            albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            albumentations.ShiftScaleRotate(shift_limit=0.0625,
                                            scale_limit=0.1,
                                            rotate_limit=5,
                                            p=0.9),
            OutsideCutout(bin_size=30, always_apply=True),
            albumentations.Normalize(mean, std, always_apply=True)
        ])

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, item):
        image = joblib.load(f"{self.pkl_file_path}/{self.img_id[item]}.pkl")
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image, [2,0,1]).astype(float) # for using torchvision model
        return {
            'image':torch.tensor(image, dtype=torch.float),
            'label': torch.tensor(self.labels[item], dtype=torch.long)
        }