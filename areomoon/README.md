# Mongochu

## Setup 

```
pip install -r areomoon/requirements.txt
```

## Image Data Preprocess
```
python src/create_fold.py --label_path ../AImongo_img/train.csv --save_path ../AImongo_img/train_folds.csv
```
```
python src/create_image_pkl.py --jpg_file_path ../AImongo_img/C1-P1_Train --pkl_file_path ../AImongo_img/train_pkl_files
```

## Training
```
mkdir src/weights
```
```
python src/train.py --fold_file ../AImongo_img/train_folds.csv  --pkl_file ../AImongo_img/train_pkl_files
```

