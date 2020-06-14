import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
import argparse
from utils import model_dispatcher, onehot
from dataset import ImageSamplerDataset
from torch.utils.data import DataLoader
from torch.optim import Adam,lr_scheduler, AdamW
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from torchtoolbox.nn import LabelSmoothingLoss

MODEL_MEAN = (0.485,0.456,0.406)
MODEL_STD = (0.229,0.224,0.225)

TRAIN_FOLDS = [0,1,2,3]
VALID_FOLDS = [4]

parser = argparse.ArgumentParser(description='Mango Defection Classification With Pytorch')

parser.add_argument('--train_file', default='../AIMango_img/train.csv', type=str,
                    help='path to input data')

parser.add_argument('--image_file', default='../AIMango_img/C1-P1_Train', type=str,
                    help='path to input data')

parser.add_argument('--image_height', default=137, type=int,
                    help='input image height')

parser.add_argument('--image_width', default=236, type=int,
                    help='input image width')

parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers for loading image')

parser.add_argument('--device', default='cuda', type=str,
                    help='device for train and eval')

parser.add_argument('--base_model', default='vgg16', type=str,
                    help='base model to use')

parser.add_argument('--lr', default=1e-4, type=float,
                    help='learning rate')

parser.add_argument('--weight_decay', default=5e-5, type=float,
                    help='regularization')

parser.add_argument('--epochs', default=3, type=int,
                    help='Number of epoch for training')

parser.add_argument('--train_batch_size', default=256, type=int,
                    help='Batch size for training')

parser.add_argument('--test_batch_size', default=128, type=int,
                    help='Batch size for training')

parser.add_argument('--test_size', default=0.2, type=float,
                    help='proportion of test size')

parser.add_argument('--random_state', default=42, type=int,
                    help='random seed for train test split')

parser.add_argument('--save_dir', default='../weights', type=str,
                    help='directory to save model')

parser.add_argument('--beta', default=1.0, type=float,
                    help='hyperparameter beta')

parser.add_argument('--cutmix_prob', default=1.0, type=float,
                    help='cutmix probability')

parser.add_argument('--binclass', default=None, type=str,
                    help='specify class for binary classification')

parser.add_argument('--nclass', default=3, type=int,
                    help='number of classes')

args = parser.parse_args()


LSLoss = LabelSmoothingLoss(3, smoothing=0.1)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def loss_fn(outputs, target):
    class_weights = torch.tensor([[1, 2, 1]]).type(torch.FloatTensor).cuda() # hardcode class weight here
    loss = nn.CrossEntropyLoss(weight=class_weights)(outputs, target)
    return loss

def focal_loss_fn(outputs, target):
    alpha = torch.tensor(1.0, dtype=torch.float64, device=torch.device('cuda'))
    gamma = torch.tensor(2.0, dtype=torch.float64, device=torch.device('cuda'))

    target = target.view(-1,1)
    logpt = nn.functional.log_softmax(outputs, dim=1)
    logpt = logpt.gather(1, target)
    logpt = logpt.view(-1)
    pt = Variable(logpt.data.exp())

    loss = -alpha * (1-pt)**gamma * logpt
    return loss.mean()

def train(dataset_size, dataloader, model, optimizer, device, loss_fn):
    model.train()
    losses = AverageMeter()

    for batch_ind, d in tqdm(enumerate(dataloader), total=dataset_size/dataloader.batch_size):
        image = d['image']
        label = d['label']
        image = image.to(device,dtype=torch.float)
        target = label.to(device, dtype=torch.long)

        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.clip(np.random.beta(args.beta, args.beta), 0.3, 0.7)
            rand_index = torch.randperm(image.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
            image[:, :, bbx1:bbx2, bby1:bby2] = image[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))
            # compute output
            outputs = model(image)
            loss = loss_fn(outputs, target_a) * lam + loss_fn(outputs, target_b) * (1. - lam)
        else:
            # compute output
            outputs = model(image)
            loss = loss_fn(outputs, target)

        losses.update(loss.item(), image.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


def evaluate(dataset_size, dataloader, model, device,loss_fn, tag):
    model.eval()
    losses = AverageMeter()
    image_pred_list = []
    image_target_list = []
    with torch.no_grad():
        for batch_ind, d in tqdm(enumerate(dataloader),total=dataset_size/dataloader.batch_size):
            image = d['image']
            label = d['label']
            image = image.to(device,dtype=torch.float)
            target = label.to(device, dtype=torch.long)
            outputs = model(image)

            loss = loss_fn(outputs, target)
            losses.update(loss.item(), image.size(0))

            pred_label = torch.argmax(outputs, dim=1)
            image_pred_list.append(pred_label)
            image_target_list.append(target)

    # Evaludation Metrics
    pred = torch.cat(image_pred_list).cpu().numpy()
    tgt = torch.cat(image_target_list).cpu().numpy()
    
    if not args.binclass:
        cfm = np.round(confusion_matrix(y_true=tgt, y_pred=pred, labels=[0,1,2]), 3)
    else:
        cfm = np.round(confusion_matrix(y_true=tgt, y_pred=pred, labels=[0,1]), 3)
    
    accu = accuracy_score(y_true=tgt,y_pred=pred)
    if tag == 'train':
        print(f'Confusion Matrix of {tag}')
        print(cfm)
        print('General Accuracy score on Train: {:5.4f}'.format(accu))
        return final_loss/counter, accu
    elif tag == 'valid':
        print(f'Confusion Matrix of {tag}')
        print(cfm)
        print('General Accuracy score on Valid: {:5.4f}'.format(accu))
        return losses.avg, accu

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_train_valid_indice(test_size=0.2, random_state=42):
    df_tr = pd.read_csv(args.train_file)
    
    indice = df_tr.index
    label = df_tr.label

    train_indice, valid_indice, train_y, valid_y= train_test_split(indice, label, test_size=test_size, random_state=random_state, stratify=label)

    return train_indice, valid_indice

def main():

    if device =='cuda':
        torch.backends.cudnn.benchmark = True #  should add to speed up the code when input array shape doesn't vary
        print('Using cudnn.benchmark.')

    model = model_dispatcher(True, base_model, nclass)
    model.to(device)

    train_size = len(pd.read_csv(train_file))
    print(train_size)

    train_dataset = ImageSamplerDataset(
        phase = 'train',
        train_file = train_file,
        image_file_path = image_file,
        image_height=image_height,
        image_width=image_width,
        mean=MODEL_MEAN,
        std=MODEL_STD,
        binclass = binclass
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler_cosine = CosineAnnealingLR(optimizer, epochs)

    if torch.cuda.device_count() > 1 :
        model = nn.DataParallel()


    for epoch in range(epochs):
        tr_loss = train(dataset_size=train_size ,dataloader=train_dataloader, model=model, optimizer=optimizer, device=device, loss_fn=focal_loss_fn)
        print(f'Epoch_{epoch+1} Train Loss:{tr_loss}')

        scheduler_cosine.step(epoch)

    torch.save(model.state_dict(), os.path.join(save_dir, f'{base_model}_on_all_epoch11.bin'))
    print('train on all is complete')




if __name__ == '__main__':
    print(args)
    main()