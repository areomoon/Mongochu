import torch
import numpy as np
import pandas as pd
import joblib
import albumentations

class ImageDataset:
    def __init__(self,fold_file, pkl_file_path, folds, image_height, image_width, mean, std):
        self.pkl_file_path = pkl_file_path
        self.fold_file = fold_file

        df = pd.read_csv(self.fold_file)
        df = df[['image_id','labels','kfold']]
        df = df[df['kfold'].isin(folds)].reset_index(drop= True)

        self.img_id = df['image_id'].apply(lambda x: x.split('.')[0]).values
        self.labels = pd.get_dummies(df['labels'].apply(lambda x: x[-1])).values

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