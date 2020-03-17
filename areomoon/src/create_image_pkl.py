import joblib
import glob
import argparse
from tqdm import tqdm
import cv2
import os

parser = argparse.ArgumentParser(description='Mango Defection Classification With Pytorch')
parser.add_argument('--jpg_file_path ', default='../AIMango_sample/sample_image', type=str,
                    help='path to input data')
parser.add_argument('--pkl_file_path', default='../AIMango_sample/pkl_files', type=str,
                    help='path to input data')
args = parser.parse_args()

# jpg_file_path = '../AIMango_sample/sample_image'
# pkl_file_path = '../AIMango_sample/pkl_files'


if __name__ == '__main__':
    if not os.path.exists(args.pkl_file_path):
        os.mkdir(args.pkl_file_path)

    files = glob.glob(os.path.join(args.jpg_file_path,'*.jpg'))
    for f in tqdm(files):
        img_id = os.path.basename(f).split('.')[0]
        img_bgr = cv2.imread(f)
        img_rgb = img_bgr[:, :, [2, 1, 0]]
        joblib.dump(img_rgb,f"{args.pkl_file_path}/{img_id}.pkl")
