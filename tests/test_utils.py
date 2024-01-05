
import sys
import os
import torch
from glob import glob
import numpy as np
from PIL import Image
import torchvision

root = "/".join(sys.path[0].split('/')[:-1])
sys.path.insert(0, root)
os.chdir(root)

from doorcam import utils 

def test_test_transform():
    image = np.random.randint(255, size=(224, 224, 3)).astype(np.uint8)
    pil_image = Image.fromarray(image)
    image_transformed = utils.test_transform(pil_image)
    assert image_transformed.dtype == torch.float

def test_train_transform():
    image = np.random.randint(255, size=(224, 224, 3)).astype(np.uint8)
    pil_image = Image.fromarray(image)
    image_transformed = utils.train_transform(pil_image)
    assert image_transformed.dtype == torch.float

def test_get_transform_train():
    transform = utils.get_transform("train")
    image = np.random.randint(255, size=(224, 224, 3)).astype(np.uint8)
    pil_image = Image.fromarray(image)
    image_transformed = transform(pil_image)
    assert image_transformed.dtype == torch.float

def test_get_transform_test():
    transform = utils.get_transform("test")
    image = np.random.randint(255, size=(224, 224, 3)).astype(np.uint8)
    pil_image = Image.fromarray(image)
    image_transformed = transform(pil_image)
    assert image_transformed.dtype == torch.float

def test_get_transform_val():
    transform = utils.get_transform("val")
    image = np.random.randint(255, size=(224, 224, 3)).astype(np.uint8)
    pil_image = Image.fromarray(image)
    image_transformed = transform(pil_image)
    assert image_transformed.dtype == torch.float

