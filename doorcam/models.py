import torch
import torch.nn as nn
import torchvision.models as models
import yaml
from torchvision.models import ResNet18_Weights, ResNet50_Weights

import aggregation as agg
from utils import get_config

config = get_config()


def get_backbone(config: dict) -> nn.Module:
    """
    Returns the backbone used for feature extraction on the image

    Args:
        config (dict): arguments loaded from the configuration file config.yaml

    Returns:
        nn.Module: a backbone model for feature extraction on image faces
    """
    image_size = config["training"]["image_size"]

    if config["training"]["backbone"] == "resnet18":
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(*(list(model.children())[:-2])).to(
            config["training"]["device"]
        )
        test_img = torch.randn(1, 3, image_size[0], image_size[1])
        feature_dim = (
            model(test_img.to(config["training"]["device"]))
            .detach()
            .cpu()
            .squeeze()
            .numpy()
            .shape
        )
        config["training"]["feature_dim"] = feature_dim
        backbone = model

    if config["training"]["backbone"] == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(*(list(model.children())[:-2])).to(
            config["training"]["device"]
        )
        test_img = torch.randn(1, 3, image_size[0], image_size[1])
        feature_dim = (
            model(test_img.to(config["training"]["device"]))
            .detach()
            .cpu()
            .squeeze()
            .numpy()
            .shape
        )
        config["training"]["feature_dim"] = feature_dim
        backbone = model

    return backbone


def get_aggregation(args: dict) -> nn.Module:
    """
    Returns a model to convert the convolutional feature maps of the feature extractor
    into a single vector representation

    Args:
        args (dict): arguments loaded from the configuration file config.yaml

    Returns:
        nn.Module: a aggregation model for feature extraction on image faces
    """

    if config["training"]["aggregation"] == "MAC":
        aggregation = agg.MAC().to(config["training"]["device"])
        return aggregation

    elif config["training"]["aggregation"] == "SPOC":
        aggregation = agg.SPoC().to(config["training"]["device"])
        return aggregation

    elif config["training"]["aggregation"] == "GEM":
        aggregation = agg.GeM().to(config["training"]["device"])
        return aggregation

    else:
        raise NotImplementedError


class FaceIDModel(nn.Module):
    """A PyTorch model for face identification.

    This model inherits from `nn.Module` and implements the forward pass
    for face identification. The constructor takes a dictionary of arguments
    which can be used to configure the model.

    Args:
        args (dict): A dictionary of arguments for configuring the model.

    Attributes:
        backbone: The nn.Module class for a convolutional feature map extractor
        aggregation: The nn.Module class that aggregates feature maps into a single vector embedding
        linear: the linear layer that projects the embedding to the required dimension sepcified in the config.yaml
    """

    def __init__(self, config: dict) -> None:
        """Initialize the FaceIDModel with the given arguments.

        Args:
            args (dict): A dictionary of arguments for configuring the model.
        """
        super().__init__()

        self.backbone = get_backbone(config)
        self.aggregation = get_aggregation(config)

        test_image = torch.randn(
            1,
            3,
            config["training"]["image_size"][0],
            config["training"]["image_size"][1],
        ).to(config["training"]["device"])
        feature_maps = self.backbone(test_image)
        embedding = self.aggregation(feature_maps).detach().cpu().flatten()
        self.linear_proj = nn.Linear(
            embedding.shape[0], config["training"]["fc_output_dim"]
        )
        self.norm = agg.L2Norm()

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Define the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor to the model.

        Returns:
            torch.Tensor: The output tensor from the model.
        """
        x = self.backbone(x)
        x = self.aggregation(x).flatten(1)
        x = self.linear_proj(x)
        x = self.norm(x)
        return x
