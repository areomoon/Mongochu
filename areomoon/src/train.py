import os
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from dataset import ImageDataset
from torch.utils.data import DataLoader
from torch.optim import Adam,lr_scheduler
from model_dispatcher import MODEL_DISPATCHER

MODEL_MEAN = (0.485,0.456,0.406)
MODEL_STD = (0.229,0.224,0.225)

TRAIN_FOLDS = [0,1,2,3]
VALID_FOLDS = [4]

parser = argparse.ArgumentParser(description='Mango Defection Classification With Pytorch')

parser.add_argument('--fold_file', default='../AIMango_sample/train_folds.csv', type=str,
                    help='path to input data')

parser.add_argument('--pkl_file', default='../AIMango_sample/pkl_files', type=str,
                    help='path to input data')

parser.add_argument('--image_height', default=137, type=int,
                    help='input image height')

parser.add_argument('--image_width', default=236, type=int,
                    help='input image width')

parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers for loading image')

parser.add_argument('--device', default='cuda', type=str,
                    help='device for train and eval')

parser.add_argument('--base_model', default='resnet34', type=str,
                    help='base model to use')

parser.add_argument('--lr', default=1e-4, type=float,
                    help='learning rate')

parser.add_argument('--epochs', default=10, type=int,
                    help='Number of epoch for training')

parser.add_argument('--train_batch_size', default=128, type=int,
                    help='Batch size for training')

parser.add_argument('--test_batch_size', default=128, type=int,
                    help='Batch size for training')

parser.add_argument('--save_dir', default='../weights', type=str,
                    help='directory to save model')

args = parser.parse_args()

def loss_fn(outputs,target):
    loss = nn.CrossEntropyLoss()(outputs, target)
    return loss


def train(dataset, dataloader, model, optimizer, device, loss_fn):
    model.train()
    for batch_ind, d in tqdm(enumerate(dataloader),total=int(len(dataset))/dataloader.batch_size):
        image = d['image']
        label = d['label']
        image = image.to(device,dtype=torch.float)
        target = label.to(device, dtype=torch.long)
        optimizer.zero_grad()
        outputs = model(image)

        loss = loss_fn(outputs,target)
        loss.backward()
        optimizer.step()


def evaluate(dataset, dataloader, model, device,loss_fn):
    model.eval()
    final_loss = 0
    counter = 0
    with torch.no_grad():
        for batch_ind, d in tqdm(enumerate(dataloader),total=int(len(dataset))/dataloader.batch_size):
            counter += 1
            image = d['image']
            label = d['label']
            image = image.to(device,dtype=torch.float)
            target = label.to(device, dtype=torch.long)
            outputs = model(image)

            loss = loss_fn(outputs,target)
            final_loss += loss
    return final_loss/counter


def main():

    if args.device =='cuda':
        torch.backends.cudnn.benchmark = True # Good optimizer when input array shape doesn't vary
        print('Using cudnn.benchmark.')

    model = MODEL_DISPATCHER[args.base_model]
    model.to(args.device)

    train_dataset = ImageDataset(
        fold_file = args.fold_file,
        pkl_file_path = args.pkl_file,
        folds=TRAIN_FOLDS,
        image_height=args.image_height,
        image_width=args.image_width,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    valid_dataset = ImageDataset(
        fold_file=args.fold_file,
        pkl_file_path=args.pkl_file,
        folds=VALID_FOLDS,
        image_height=args.image_height,
        image_width=args.image_width,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    optimizer = Adam(model.parameters(),lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.3)

    if torch.cuda.device_count() > 1 :
        model = nn.DataParallel()

    for epoch in range(args.epochs):
        train(dataset=train_dataset,dataloader=train_dataloader,model=model,optimizer=optimizer,device=args.device,loss_fn=loss_fn)
        val_loss = evaluate(dataset=valid_dataset, dataloader=valid_dataloader, model=model, device=args.device,loss_fn=loss_fn)
        print(f'Epoch_{epoch+1} Loss:{val_loss}')
        scheduler.step(val_loss)
        torch.save(model.state_dict(),os.path.join(args.save_dir,f'{args.base_model}_fold_{VALID_FOLDS[0]}.bin'))

if __name__ == '__main__':
    main()