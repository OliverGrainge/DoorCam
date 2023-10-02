import yaml
from torchvision import transforms

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


train_transforms = transforms.Compose([transforms.Resize(tuple(config["training"]["image_size"])), transforms.ToTensor()])

test_transforms = transforms.Compose([transforms.Resize(tuple(config["training"]["image_size"])), transforms.ToTensor()])
