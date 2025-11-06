import os
import torch
from torchvision import datasets, transforms
from PIL import Image

IMG_SIZE  = 224 


# resize with padding - function approach ===========================
from Datasets.image_resize_with_padding import resize_with_padding

# wrap function with IMG_SIZE
from functools import partial
resize_fn = partial(resize_with_padding, img_size=IMG_SIZE)

# full transform pipeline
tsfm = transforms.Compose([
    # new resize fn
    transforms.Lambda(resize_fn),

    # others all the same
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10, fill=0),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# resize with padding - function approach ===========================
from Datasets.image_resize_with_padding import ResizeWithPadding

# full transform pipeline
tsfm = transforms.Compose([
    # new resize fn
    ResizeWithPadding(224),

    # others all the same
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10, fill=0),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])