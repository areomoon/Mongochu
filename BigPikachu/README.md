# Mongochu

## Get Started
### Train

- Train **multi-class** ```vgg16``` model.
```
python train_cutmix_with_sampler.py \
--train_file /kaggle/input/aimongo/AImongo_img/train.csv \ 
--image_file /kaggle/input/aimongo/AImongo_img/C1-P1_Train \
--lr 1e-4  \
--epochs 2 \
--image_height 224 \
--image_width 224 \
--train_batch_size 64 \
--base_model vgg16 \
--nclass 3 \
--beta 1 \
--cutmix_prob 0.1 \
--test_size 0.2 \
--random_state 2020
```

- Train **binary(one-vs-rest)** ```vgg16``` model.
```
python train_cutmix_with_sampler.py \
--train_file /kaggle/input/aimongo/AImongo_img/train.csv \ 
--image_file /kaggle/input/aimongo/AImongo_img/C1-P1_Train \
--lr 1e-4  \
--epochs 2 \
--image_height 224 \
--image_width 224 \
--train_batch_size 64 \
--base_model vgg16 \
--binclass A \
--nclass 2 \
--beta 1 \
--cutmix_prob 0.1 \
--test_size 0.2 \
--random_state 2020
```

### Inference
- Output predicted classes for samples in Dev.  
Save the ```vgg16_eval_submission.csv``` file under working directory ```BigPikachu/src/```.  
**Notes: modify ```nclass``` for binary/multiclass.**
```
python eval.py \
--image_file /kaggle/input/aimongo/AImongo_img/C1-P1_Dev \  
--model_weights vgg16_fold_4.bin \
--image_height 224 \
--image_width 224 \
--base_model vgg16 \
--nclass 3
```
- Output predicted probabilities of each classes for samples in Dev.  
Save the ```vgg16_prob_v1.csv``` file under working directory ```BigPikachu/src/```.  
**Notes: currently don't support binary classification.**
```
python eval_prob.py \
--image_file /kaggle/input/aimongo/AImongo_img/C1-P1_Dev \
--model_weights vgg16_fold_4.bin \
--image_height 224 \
--image_width 224 \
--base_model vgg16 \
--nclass 3 \
--output_name vgg16_prob_v1
```

### Blending
Put all the files of predicted probabilities for each class under the ```folder_path```.  
Then chose hard or soft voting for blending.  
Please refer to [sklearn.VotingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)  
- hard blending
```
python blending.py \
--folder_path ./pred \
--voting hard \
--output_name hard_blend
```

- soft blending
```
python blending.py \
--folder_path ./pred \
--voting soft \
--output_name soft_blend
```
