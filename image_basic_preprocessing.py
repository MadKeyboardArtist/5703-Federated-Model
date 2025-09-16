import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt

IMG_SIZE  = 224 
TRAIN_DIR = "image_dataset/retina_extracted/train"
VAL_DIR   = "image_dataset/retina_extracted/val"
TEST_DIR  = "image_dataset/retina_extracted/test"

# Preprocessing process
tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness = 0.1,
        contrast = 0.1,
        saturation = 0.1,
        hue = 0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform = tfms)
val_dataset   = datasets.ImageFolder(VAL_DIR  , transform = tfms)
test_dataset  = datasets.ImageFolder(TEST_DIR , transform = tfms)

# use it like:
'''
from image_basic_preprocessing import tfms
train_dataset = datasets.ImageFolder("path/to/train", transform = tfms)
'''


