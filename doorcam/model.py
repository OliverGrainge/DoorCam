import torch 
import torch.nn as nn
import torchvision 
import torchvision.models as models




def get_backbone(args: dict) -> nn.Module:
    """
    Returns the backbone used for feature extraction on the image

    Args:
        args (dict): arguments loaded from the configuration file config.yaml

    Returns:
        nn.Module: a backbone model for feature extraction on image faces
    """
    image_size = args["training"]["image_size"]

    if args["training"]["backbone"] == "resnet18":
        model = models.resnet18(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-2])).to(args["training"]["device"])
        test_img = torch.randn(1, 3, image_size[0], image_size[1])
        feature_dim = model(test_img.to(args["training"]["device"])).detach().cpu().squeeze().numpy().shape
        args["training"]["feature_dim"] = feature_dim
        backbone = model

    if args["training"]["backbone"] == "resnet50":
        model = models.resnet18(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-2])).to(args["training"]["device"])
        test_img = torch.randn(1, 3, image_size[0], image_size[1])
        feature_dim = model(test_img.to(args["training"]["device"])).detach().cpu().squeeze().numpy().shape
        args["training"]["feature_dim"] = feature_dim
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
    return None 



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

    def __init__(self, args: dict) -> None:
        """Initialize the FaceIDModel with the given arguments.

        Args:
            args (dict): A dictionary of arguments for configuring the model.
        """
        super().__init__()

        self.backbone = get_backbone(args)
        self.aggregation = get_aggregation(args)
        self.linear = nn.Linear(args["feature_dim"], args["fc_ouput_dim"])


    def forward(self, x: torch.tensor) -> torch.tensor:
        """Define the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor to the model.

        Returns:
            torch.Tensor: The output tensor from the model.
        """
        return None 
