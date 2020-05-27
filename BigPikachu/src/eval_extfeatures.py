import os
from itertools import chain
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import argparse
from utils import model_dispatcher
from dataset import ImageTestDataset
from torch.utils.data import DataLoader

MODEL_MEAN = (0.485,0.456,0.406)
MODEL_STD = (0.229,0.224,0.225)

parser = argparse.ArgumentParser(description='Mango Defection Classification With Pytorch')

parser.add_argument('--image_file', default = None, type=str,
                    help='path to input data')

parser.add_argument('--image_height', default=137, type=int,
                    help='input image height')

parser.add_argument('--image_width', default=236, type=int,
                    help='input image width')

parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers for loading image')

parser.add_argument('--device', default='cuda', type=str,
                    help='device for train and eval')

parser.add_argument('--base_model', default='vgg16_eval', type=str,
                    help='base model to use')

parser.add_argument('--lr', default=1e-4, type=float,
                    help='learning rate')

parser.add_argument('--test_batch_size', default=128, type=int,
                    help='Batch size for training')

parser.add_argument('--save_dir', default='../weights', type=str,
                    help='directory to save model')

parser.add_argument('--model_weights', default=None, type=str,
                    help='model weights to load')

parser.add_argument('--nclass', default=3, type=int,
                    help='number of classes')

args = parser.parse_args()

def main():
    model = model_dispatcher(False, args.base_model, args.nclass)
    model.to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir,args.model_weights)))
    model.eval()
    print(f'Loading pretrained model: {args.base_model} for eval')

    test_dataset = ImageTestDataset(
        file_path = args.image_file,
        image_height=args.image_height,
        image_width=args.image_width,
        mean = MODEL_MEAN,
        std = MODEL_STD
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    image_id_list = []
    image_feat_list = []

    with torch.no_grad():
        for batch_id, d in enumerate(tqdm(test_dataloader)):
            image = d['image']
            img_id = d['image_id']

            image = image.to(args.device, dtype=torch.float)
            img_features = model.model._features(image)

            image_id_list.append(img_id)
            image_feat_list.append(img_features.cpu().numpy())
    print(image_feat_list[0].shape)
    preds = torch.cat(image_feat_list)

    ids = list(chain(*image_id_list))

    sub = pd.DataFrame({'image_ids':ids, 'features':preds})
    sub.to_csv(f'{args.base_model}_imagefeatures.csv',index=False)

if __name__ == '__main__':
    main()