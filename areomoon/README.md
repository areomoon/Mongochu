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

## Image Augmentation Illustration
WIP

## File Structure
 ./   
├── areomoon       
│    └── AImongo_img    
│       └── C1-P1-Train  
│       └──C1-P1-Dev  
│       └──..      
│   └── src   
│   └── weights  
│   └── requirements.txt
   

## Image Data Preprocess

Under working directory ```areomoon/src/``` run the following code:

- 1 Stratified split the whole image dataset into 5 folds (can adjust the code) and create the label file(*.csv*)
```
python create_fold.py --label_path ../AImongo_img/train.csv --save_path ../AImongo_img/train_folds.csv
```

- 2 Transfer the raw image to *.pkl*  file for training later 

```
python create_image_pkl.py --jpg_file_path ../AImongo_img/C1-P1_Train --pkl_file_path ../AImongo_img/train_pkl_files
```

## Train

Create ```src/weights``` folder to save model weights
```
mkdir src/weights
```

Under working directory ```areomoon/src/``` run the following code:
```
python train.py --fold_file ../AImongo_img/train_folds.csv  --pkl_file ../AImongo_img/train_pkl_files
```

## Inference

WIP
