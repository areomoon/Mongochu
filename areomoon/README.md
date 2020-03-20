# Mongochu

## Setup 
```
pip install -r areomoon/requirements.txt
```

### Dependency
Base
- numpy
- pandas
- iterative-stratification

Image Process
- opencv-python
- albumentations

Model
- torch
- pretrainedmodels


## Image Data Preprocess

Move to working directory ```areomoon/src/``` run the following code:

1.Stratified split the whole image dataset into 5 folds (can adjust the code) and create the  
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
python src/train.py --fold_file ../AImongo_img/train_folds.csv  --pkl_file ../AImongo_img/train_pkl_files
```

