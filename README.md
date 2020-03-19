# Mongochu

Hi all, please feel free to add notes

## Setup 

```
pip install -r requirements.txt
```
```
python create_fold.py --label_path ../AImongo_img/train.csv --save_path ../AImongo_img/train_folds.csv
```
```
python create_image_pkl.py --jpg_file_path ../AImongo_img/C1-P1_Train --pkl_file_path ../AImongo_img/train_pkl_files
```

## Training
```
mkdir src/weights
```
```
python train.py --fold_file ../AImongo_img/train_folds.csv  --pkl_file ../AImongo_img/train_pkl_files
```

