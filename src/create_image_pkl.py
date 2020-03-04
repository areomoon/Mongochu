import joblib
import glob
from tqdm import tqdm
import cv2
import os


jpg_file_path = '../AIMango_sample/sample_image'
pkl_file_path = '../AIMango_sample/pkl_files'


if __name__ == '__main__':
    if not os.path.exists(pkl_file_path):
        os.mkdir(pkl_file_path)

    files = glob.glob(os.path.join(jpg_file_path,'*.jpg'))
    for f in tqdm(files):
        img_id = os.path.basename(f).split('.')[0]
        img_bgr = cv2.imread(f)
        img_rgb = img_bgr[:, :, [2, 1, 0]]
        joblib.dump(img_rgb,f"{pkl_file_path}/{img_id}.pkl")
