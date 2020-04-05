import torch
from torch.nn import functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations
import cv2

MODEL_MEAN = (0.485,0.456,0.406)
MODEL_STD = (0.229,0.224,0.225)

class CAM:
    def __init__(self, model,image_height=224, image_width=224, mean=MODEL_MEAN, std=MODEL_STD):
        self.model = model
        self.features = None
        self.prediction = None
        self.image_height=image_height
        self.image_width=image_width
        self.mean=mean
        self.std =std
        self.id_map = {0:'A',1:'B',2:'C'}

        weight, bias = list(self.model.l0.parameters())
        self.weight = weight.detach().numpy()  # (3, 512)

        del weight, bias

    def _forward(self, img):
        with torch.no_grad():
            features = self.model.model.features(img)  # (1, c, h, w)
            conv_features = F.adaptive_avg_pool2d(features, 1).reshape(1, -1)
            logits = self.model.l0(conv_features)  # (1, n_classes)
            prediction = F.softmax(logits, dim=-1)

        self.features = features.numpy().squeeze()  # (c, w, h)
        self.prediction = prediction.numpy().squeeze()  # (n_classes)

    def _get_class_idx(self, i):
        class_idx = self.prediction.argsort()
        class_idx = class_idx[-i]

        return class_idx

    def _generate_heatmap(self, class_idx,height,width):
        weight_c = self.weight[class_idx, :]  # (c,)
        weight_c = weight_c.reshape((-1, 1, 1))  # (c, 1, 1)

        heatmap = np.sum(weight_c * self.features, axis=0)  # (h, w)
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize between 0-1
        heatmap = np.uint8(heatmap * 255)
        heatmap = Image.fromarray(heatmap)
        heatmap = heatmap.resize((height, width), Image.ANTIALIAS)
        heatmap = np.array(heatmap)

        return heatmap

    def plot_image_heatmap(self, img_path, top=3):
        img_bgr = cv2.imread(img_path)
        height = img_bgr.shape[1]
        width  = img_bgr.shape[0]
        img= img_bgr[:, :, [2, 1, 0]]
        aug = albumentations.Compose([
            albumentations.Resize(self.image_height, self.image_width, always_apply=True),
            albumentations.Normalize(self.mean, self.std, always_apply=True),

        ])
        img_tensor = aug(image=np.array(img))['image']
        img_tensor = np.transpose(img_tensor, [2, 0, 1]).astype(float)
        img_tensor = torch.unsqueeze(torch.from_numpy(img_tensor).float(),0)
        self._forward(img_tensor)

        plt.imshow(img)
        plt.axis('off')

        cols = top + 1
        plt.figure(figsize=(12, 12 * cols))
        for i in range(cols):
            if i == 0:
                plt.subplot(1, cols, i + 1)
                plt.imshow(img, alpha=1.0)
                plt.title('Original image')
                plt.axis('off')
            else:
                class_idx = self._get_class_idx(i)
                label = self.id_map[class_idx]
                proba = self.prediction[class_idx]
                heatmap = self._generate_heatmap(class_idx,height,width)

                plt.subplot(1, cols, i + 1)
                plt.imshow(img, alpha=1.0)
                plt.imshow(heatmap, cmap='jet', alpha=0.5)
                plt.title('{} ({:.3f})'.format(label, proba))
                plt.axis('off')
