import os
from itertools import chain
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import argparse
from utils import model_dispatcher
from dataset import ImageTTADataset
from torch.utils.data import DataLoader
import albumentations

parser = argparse.ArgumentParser(description='Mango Defection Classification With Pytorch')

parser.add_argument('--image_file', default = '../BigPikachu/AImongo_img/C1-P1_Dev', type=str,
                    help='path to input data')

parser.add_argument('--image_height', default=224, type=int,
                    help='input image height')

parser.add_argument('--image_width', default=224, type=int,
                    help='input image width')

parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers for loading image')

parser.add_argument('--device', default='cpu', type=str,
                    help='device for train and eval')

parser.add_argument('--base_model', default='vgg16', type=str,
                    help='base model to use')

parser.add_argument('--test_batch_size', default=128, type=int,
                    help='Batch size for training')

parser.add_argument('--save_dir', default='./weights', type=str,
                    help='directory to save model')

parser.add_argument('--model_weights', default='vgg16_fold_4_new_aug.bin', type=str,
                    help='model weights to load')

parser.add_argument('--nclass', default=3, type=int,
                    help='number of classes')

parser.add_argument('--num_tta', default=8, type=int,
                    help='number of TTA')

parser.add_argument('--output_name', default='TTA_submission', type=str,
                    help='output name of submission')

args = parser.parse_args()

data_transforms = albumentations.Compose([
    albumentations.Resize(args.image_height, args.image_width),
    albumentations.RandomRotate90(p=0.5),
    albumentations.Transpose(p=0.5),
    albumentations.Flip(p=0.5),
    albumentations.OneOf([
        albumentations.CLAHE(clip_limit=2), albumentations.IAASharpen(), albumentations.IAAEmboss(), 
        albumentations.RandomBrightness(), albumentations.RandomContrast(),
        albumentations.JpegCompression(), albumentations.Blur(), albumentations.GaussNoise()], p=0.5), 
    albumentations.HueSaturationValue(p=0.5), 
    albumentations.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=45, p=0.5),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    albumentations.ToFloat()
    ])

data_transforms_test = albumentations.Compose([
    albumentations.Resize(args.image_height, args.image_width),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    # albumentations.ToFloat()
    ])

data_transforms_tta0 = albumentations.Compose([
    albumentations.Resize(args.image_height, args.image_width),
    albumentations.RandomRotate90(p=0.5),
    albumentations.Transpose(p=0.5),
    albumentations.Flip(p=0.5),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    # albumentations.ToFloat()
    ])

data_transforms_tta1 = albumentations.Compose([
    albumentations.Resize(args.image_height, args.image_width),
    albumentations.RandomRotate90(p=1),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    # albumentations.ToFloat()
    ])

data_transforms_tta2 = albumentations.Compose([
    albumentations.Resize(args.image_height, args.image_width),
    albumentations.Transpose(p=1),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    # albumentations.ToFloat()
    ])

data_transforms_tta3 = albumentations.Compose([
    albumentations.Resize(args.image_height, args.image_width),
    albumentations.Flip(p=1),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    # albumentations.ToFloat()
    ])

def main():
    model = model_dispatcher(False, args.base_model, args.nclass)
    model.to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir,args.model_weights)))

    # checkpoint = torch.load(os.path.join(args.save_dir,args.model_weights), map_location=args.device)
    # model.load_state_dict(checkpoint)

    model.eval()
    print(f'Loading pretrained model: {args.base_model} for eval')

    for num_tta in range(args.num_tta):
        if num_tta == 0:
            test_dataset = ImageTTADataset(file_path = args.image_file, transform=data_transforms_test)
            test_dataloader =DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

        elif num_tta == 1:
            test_dataset = ImageTTADataset(file_path = args.image_file, transform=data_transforms_tta1)
            test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

        elif num_tta == 2:
            test_dataset = ImageTTADataset(file_path = args.image_file, transform=data_transforms_tta2)
            test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

        elif num_tta == 3:
            test_dataset = ImageTTADataset(file_path = args.image_file, transform=data_transforms_tta3)
            test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

        elif num_tta < 8:
            test_dataset = ImageTTADataset(file_path = args.image_file, transform=data_transforms_tta0)
            test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

        else:
            test_dataset = ImageTTADataset(file_path = args.image_file, transform=data_transforms)
            test_dataloader =DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)


        image_id_list = []
        image_pred_list = []

        with torch.no_grad():
            for d in test_dataloader:
                image = d['image']
                img_id = d['image_id']

                image = image.to(args.device, dtype=torch.float)
                outputs = model(image)
                pred_prob = torch.nn.Softmax(dim=1)(outputs)

                image_id_list.append(img_id)
                image_pred_list.append(pred_prob/args.num_tta)
            
            if num_tta == 0:
                ids = list(chain(*image_id_list))
                preds = torch.cat(image_pred_list).cpu().numpy()

            else:
                preds_tmp = torch.cat(image_pred_list).cpu().numpy()
                preds += preds_tmp

        print(num_tta)

    df_pred = pd.DataFrame(preds, columns=['A', 'B', 'C'])
    df_id = pd.DataFrame(ids, columns=['image_ids'])
    sub = pd.concat([df_id, df_pred], axis=1)
    sub.to_csv(f'{args.output_name}.csv',index=False)

if __name__ == '__main__':
    main()