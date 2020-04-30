import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Mango Defection Classification With Pytorch')

parser.add_argument('--train_path', default='../AImongo_img/train.csv', type=str,
                    help='path to train data')

parser.add_argument('--dev_path', default='../AImongo_img/dev.csv', type=str,
                    help='path to dev data')

parser.add_argument('--train_prefix', default='C1-P1_Train', type=str,
                    help='prefix of train data')

parser.add_argument('--dev_prefix', default='C1-P1_Dev', type=str,
                    help='prefix of dev data')

parser.add_argument('--export_path', default='../AImongo_img/all_data.csv', type=str,
                    help='path to export')

args = parser.parse_args()


if __name__ == '__main__':

    train = pd.read_csv(args.train_path)
    train['image_id'] = args.train_prefix + '/' + train['image_id']

    dev = pd.read_csv(args.dev_path)
    dev['image_id'] = args.dev_prefix + '/'  +dev['image_id']

    df = pd.concat([train, dev], axis=0)
    df = df.reset_index(drop=True)

    df.to_csv(args.export_path, index=False)