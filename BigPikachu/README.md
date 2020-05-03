---
tags: AI Mongo
---
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
├── BigPikachu       
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

Train **multi-class** ```vgg16``` model.
```python
!python train_cutmix_with_sampler.py \
--train_file /kaggle/input/aimongo/AImongo_img/train.csv \ 
--image_file /kaggle/input/aimongo/AImongo_img/C1-P1_Train \
--lr 1e-4  \
--epochs 2 \
--image_height 224 \
--image_width 224 \
--train_batch_size 64 \
--base_model vgg16 \
--beta 1 \
--cutmix_prob 0.1 \
--test_size 0.2 \
--random_state 2020
```

Train **binary(one-vs-rest)** ```vgg16``` model.
```python
!python train_cutmix_with_sampler.py \
--train_file /kaggle/input/aimongo/AImongo_img/train.csv \ 
--image_file /kaggle/input/aimongo/AImongo_img/C1-P1_Train \
--lr 1e-4  \
--epochs 2 \
--image_height 224 \
--image_width 224 \
--train_batch_size 64 \
--base_model vgg16_binary \
--binclass A \
--beta 1 \
--cutmix_prob 0.1 \
--test_size 0.2 \
--random_state 2020
```

## Inference
Output predicted classes for samples in Dev.  
Save the ```vgg16_eval_submission.csv``` file under working directory ```BigPikachu/src/```
```python
!python eval.py \
--image_file /kaggle/input/aimongo/AImongo_img/C1-P1_Dev \  
--model_weights vgg16_fold_4.bin \
--image_height 224 \
--image_width 224 \
--base_model vgg16_eval
```
Output predicted probabilities of each classes for samples in Dev.  
Save the ```vgg16_prob_v1.csv``` file under working directory ```BigPikachu/src/```
```python
!python eval_prob.py \
--image_file /kaggle/input/aimongo/AImongo_img/C1-P1_Dev \
--model_weights vgg16_fold_4.bin \
--image_height 224 \
--image_width 224 \
--base_model vgg16_eval \
--output_name vgg16_prob_v1
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
- image_width = 224
- image_height = 224
- epochs = 3
- batch_size = 64    

Accuracy on Dev : 0.77125
``` 