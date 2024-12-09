import zipfile
import sys
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class TrafficSignDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class TrafficSignClassifier:
    def __init__(self, data_dir, test_path, img_height=224, img_width=224):
        self.data_dir = data_dir
        self.test_path = test_path
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=43)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def evaluate_model(self):
        test = pd.read_csv(os.path.join(self.data_dir, 'Test.csv'))
        labels = test["ClassId"].values
        imgs = test["Path"].values
        test_image_paths = [os.path.join(self.test_path, img.replace('Test/', '')) for img in imgs]

        transform = transforms.Compose([
            transforms.Resize((self.IMG_HEIGHT, self.IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.feature_extractor.image_mean, std=self.feature_extractor.image_std)
        ])

        test_dataset = TrafficSignDataset(test_image_paths, labels, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for images, _ in tqdm(test_loader):
                images = images.to(self.device)
                outputs = self.model(images).logits
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)

        print(f'Test Accuracy: {accuracy_score(labels, all_preds)}')
        print(classification_report(labels, all_preds))
        cm = confusion_matrix(labels, all_preds)
        plt.figure(figsize=(25, 25))
        sns.heatmap(cm, annot=True)
        plt.show()

if __name__ == "__main__":
    classifier = TrafficSignClassifier(data_dir='../dataset', test_path='../dataset/Test')
    classifier.evaluate_model()
