import os
import glob
from itertools import chain
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import argparse
from dataset import ImageTestDataset
from model_dispatcher import MODEL_DISPATCHER
from torch.utils.data import DataLoader

MODEL_MEAN = (0.485,0.456,0.406)
MODEL_STD = (0.229,0.224,0.225)

parser = argparse.ArgumentParser(description='Mango Defection Classification With Pytorch')

parser.add_argument('--image_file', default = None, type=str,
                    help='path to input data')

parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers for loading image')

parser.add_argument('--device', default='cuda', type=str,
                    help='device for train and eval')

parser.add_argument('--test_batch_size', default=128, type=int,
                    help='Batch size for training')

parser.add_argument('--save_dir', default='../weights', type=str,
                    help='directory to save model')

args = parser.parse_args()

def get_image_size(base_model):
    if base_model == 'vgg16':
        return 224, 224

    elif base_model == 'se_resnext101_32x4d':
        return 224, 224

def main():
    model_list = glob.glob(os.path.join(args.save_dir, '*.bin'))

    for model_weight in model_list:
        base_model = model_weight.split('_')[0]
        image_height, image_width = get_image_size(base_model)

        model = MODEL_DISPATCHER[base_model]
        model.to(args.device)
        model.load_state_dict(torch.load(os.path.join(args.save_dir, model_weight)))
        model.eval()
        print(f'Loading pretrained model: {base_model} for eval')

        test_dataset = ImageTestDataset(
            file_path = args.image_file,
            image_height = image_height,
            image_width = image_width,
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
        image_pred_list = []

        with torch.no_grad():
            for batch_id, d in enumerate(tqdm(test_dataloader)):
                image = d['image']
                img_id = d['image_id']

                image = image.to(args.device, dtype=torch.float)
                outputs = model(image)

                pred_label = torch.argmax(outputs, dim=1)
                image_id_list.append(img_id)
                image_pred_list.append(pred_label)

        preds = torch.cat(image_pred_list).cpu().numpy()

        ids = list(chain(*image_id_list))

        sub = pd.DataFrame({'image_ids':ids, 'labels':preds})
        sub.to_csv(f'{base_model}_submission.csv',index=False)

if __name__ == '__main__':
    main()