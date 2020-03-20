import numpy as np
import matplotlib.pyplot as plt
from dataset import ImageExpDataset

dataset = ImageExpDataset(
    fold_file='AIMango_sample/train_folds.csv',
    pkl_file_path='AIMango_sample/pkl_files',
    folds=[0,1],
    image_height=137,
    image_width=236,
    mean=(0.485,0.456,0.406),
    std=(0.229,0.224,0.225)
)

print(len(dataset))

idx = range(11,20)
for i in idx:
    img = dataset[i]['image']
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1,2,0)))
    plt.show()