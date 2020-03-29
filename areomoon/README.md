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
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── AImongo_img      
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ C1-P1-Train  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──C1-P1-Dev  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──..      
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── src   
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── weights  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── requirements.txt
   

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
Run the following code and save the ```submission.csv``` file in working directory ```areomoon/src/```
```
python eval.py --image_file ../AImongo_img/C1-P1_Dev  --model_weights resnet34_fold_4.bin
```

## Evaluation 

Refer to ```areomoon/eval_metrics.ipynb```
   
#### Model Experiment and Arguments 
- Resnet34  
```
- image_width = 236
- image_height = 137
- epochs = 2
- batch_size = 256  

Accuracy on Dev :0.74375
``` 
 
- VGG16 
```
- mean=[0.485, 0.456, 0.406]
- std=[0.229, 0.224, 0.225]
- image_width = 224
- image_height = 224
- epochs = 3
- batch_size = 64    

Accuracy on Dev : 0.77125
``` 

- SE-ResNext101-32x4d
```
- mean=[0.485, 0.456, 0.406]
- std=[0.229, 0.224, 0.225]
- image_width = 224
- image_height = 224
- epochs = 1
- batch_size = 64    

Accuracy on Dev : 0.7925
``` 