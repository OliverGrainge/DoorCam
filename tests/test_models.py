import sys
import os
import torch
from glob import glob
import numpy as np
from PIL import Image
import torchvision
from torch import nn
import pytest

root = "/".join(sys.path[0].split('/')[:-1])
sys.path.insert(0, root)
os.chdir(root)


import doorcam.models as models
import doorcam.utils as utils

def test_backbone(config):
    config["training"]["device"] = "cpu"
    backbone = models.get_backbone(config)
    assert isinstance(backbone, nn.Module)


@pytest.mark.parametrize(
    "model_type",
    [
        "resnet18",
        "resnet50"
    ]
)
def test_backbone_features(model_type, config):
    config["training"]["device"] = "cpu"
    config["training"]["backbone"] = model_type
    backbone = models.get_backbone(config)
    image = np.random.randint(255, size=(224, 224, 3)).astype(np.uint8)
    pil_image = Image.fromarray(image)
    transformed_image = utils.test_transform(pil_image)
    features = backbone(transformed_image[None, :])
    assert features.ndim == 4
    assert features.dtype == torch.float
    assert features.shape[0] == 1


@pytest.mark.parametrize(
    "aggregation_type",
    [
        "MAC",
        "SPOC",
        "GEM",
    ]
)
def test_get_aggregation(aggregation_type, config):
    config["training"]["aggregation"] = aggregation_type
    config["training"]["device"] = "cpu"
    fn = models.get_aggregation(config)
    features = torch.randn(10, 128, 7, 7)
    agg_features = fn(features)
    assert agg_features.ndim == 2
    assert agg_features.shape[0] == 10
    assert agg_features.dtype == torch.float


@pytest.mark.parametrize(
    "aggregation_type, backbone_type",
    [
        ("MAC", "resnet18"),
        ("SPOC", "resnet18"),
        ("GEM", "resnet18"),
        ("MAC", "resnet50"),
        ("SPOC", "resnet50"),
        ("GEM", "resnet50"),
    ]
)
def test_faceid_model(backbone_type, aggregation_type, config):
    config["training"]["aggregation"] == aggregation_type
    config["training"]["backbone"] == backbone_type
    config["training"]["device"] = "cpu"
    model = models.FaceIDModel(config)
    assert isinstance(model, nn.Module)



@pytest.mark.parametrize(
    "aggregation_type, backbone_type",
    [
        ("MAC", "resnet18"),
        ("SPOC", "resnet18"),
        ("GEM", "resnet18"),
        ("MAC", "resnet50"),
        ("SPOC", "resnet50"),
        ("GEM", "resnet50"),
    ]
)
def test_faceid_model_features(backbone_type, aggregation_type, config):
    config["training"]["aggregation"] == aggregation_type
    config["training"]["backbone"] == backbone_type
    config["training"]["device"] = "cpu"
    model = models.FaceIDModel(config)
    image = np.random.randint(255, size=(224, 224, 3)).astype(np.uint8)
    pil_image = Image.fromarray(image)
    transformed_image = utils.test_transform(pil_image)
    features = model(transformed_image[None, :])
    assert features.ndim == 2
    assert features.shape[0] == 1
    assert features.dtype == torch.float
    assert torch.allclose(torch.norm(features), torch.tensor(1.0))





    


