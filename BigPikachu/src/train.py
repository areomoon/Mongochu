import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from dataset import ImageExpDataset
from torch.utils.data import DataLoader
from torch.optim import Adam,lr_scheduler
from model_dispatcher import MODEL_DISPATCHER
from sklearn.metrics import confusion_matrix,accuracy_score
from torchtoolbox.nn import LabelSmoothingLoss

MODEL_MEAN = (0.485,0.456,0.406)
MODEL_STD = (0.229,0.224,0.225)

TRAIN_FOLDS = [0,1,2,3]
VALID_FOLDS = [4]

parser = argparse.ArgumentParser(description='Mango Defection Classification With Pytorch')

parser.add_argument('--fold_file', default='../AIMango_img/train_folds.csv', type=str,
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

parser.add_argument('--save_dir', default='../weights', type=str,
                    help='directory to save model')

args = parser.parse_args()


LSLoss = LabelSmoothingLoss(3, smoothing=0.3)

def loss_fn(outputs,target):
    loss = nn.CrossEntropyLoss()(outputs, target)
    return loss


def train(dataset, dataloader, model, optimizer, device, loss_fn):
    model.train()
    final_loss = 0
    counter = 0
    for batch_ind, d in tqdm(enumerate(dataloader),total=int(len(dataset))/dataloader.batch_size):
        counter += 1
        image = d['image']
        label = d['label']
        image = image.to(device,dtype=torch.float)
        target = label.to(device, dtype=torch.long)
        optimizer.zero_grad()
        outputs = model(image)

        loss = loss_fn(outputs,target)
        final_loss += loss

        loss.backward()
        optimizer.step()       
    return final_loss/counter


def evaluate(dataset, dataloader, model, device,loss_fn, tag):
    model.eval()
    final_loss = 0
    counter = 0
    image_pred_list = []
    image_target_list = []
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

            pred_label = torch.argmax(outputs, dim=1)
            image_pred_list.append(pred_label)
            image_target_list.append(target)

    # Evaludation Metrics
    pred = torch.cat(image_pred_list).cpu().numpy()
    tgt = torch.cat(image_target_list).cpu().numpy()
    cfm = np.round(confusion_matrix(y_true=tgt,y_pred=pred,labels=[0,1,2]),3)
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
        return final_loss/counter, accu


def main():

    if args.device =='cuda':
        torch.backends.cudnn.benchmark = True #  should add to speed up the code when input array shape doesn't vary
        print('Using cudnn.benchmark.')

    model = MODEL_DISPATCHER[args.base_model]
    model.to(args.device)
    # print(f'Loading pretrained model: {args.base_model}')

    train_dataset = ImageExpDataset(
        fold_file = args.fold_file,
        image_file_path = args.image_file,
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

    valid_dataset = ImageExpDataset(
        fold_file=args.fold_file,
        image_file_path=args.image_file,
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

    optimizer = Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.3)

    if torch.cuda.device_count() > 1 :
        model = nn.DataParallel()

    val_accu_benchmark = 0.34
    val_loss_list = []
    val_accu_list = []
    tr_loss_list = []
    # tr_accu_list = []
    best_epoch = 0
    for epoch in range(args.epochs):
        tr_loss = train(dataset=train_dataset,dataloader=train_dataloader,model=model,optimizer=optimizer,device=args.device,loss_fn= LSLoss)
        # tr_loss, tr_accu = evaluate(dataset=train_dataset, dataloader=train_dataloader, model=model, device=args.device,loss_fn=loss_fn, tag='train')
        val_loss, val_accu = evaluate(dataset=valid_dataset, dataloader=valid_dataloader, model=model, device=args.device,loss_fn=loss_fn, tag='valid')
        print(f'Epoch_{epoch+1} Train Loss:{tr_loss}')
        print(f'Epoch_{epoch+1} Valid Loss:{val_loss}')
        scheduler.step(val_loss)
        
        tr_loss_list.append(tr_loss)
        # tr_accu_list.append(tr_accu)       
        val_loss_list.append(val_loss)
        val_accu_list.append(val_accu)
        if val_accu > val_accu_benchmark:
            best_epoch = epoch+1
            print(f'save {args.base_model} model on epoch {epoch+1}')
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'{args.base_model}_fold_{VALID_FOLDS[0]}.bin'))
            val_accu_benchmark = val_accu
    print(f'Save the best model on epoch {best_epoch}')

    stored_metrics = {'train': {
                                'tr_loss_list': tr_loss_list
                                # , 'tr_accu_list': tr_accu_list
                            },
                    'valid': {
                                'val_loss_list': val_loss_list, 'val_accu_list': val_accu_list
                             }
                  }

    # pickle a variable to a file
    file = open(os.path.join(args.save_dir, 'stored_metrics.pickle'), 'wb')
    pickle.dump(stored_metrics, file)
    file.close()

if __name__ == '__main__':
    print(args)
    main()