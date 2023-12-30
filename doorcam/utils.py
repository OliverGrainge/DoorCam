import yaml
from torchvision import transforms
from pathlib import Path
import logging
import os
import logging.config
from rich.logging import RichHandler



def get_config() -> dict:
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)

def get_logger():
    import logger
    LOGS_DIR = Path(os.path.dirname(os.path.abspath(__file__)), "logs")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.config.fileConfig(Path(LOGS_DIR, "logging.config"))
    logger = logging.getLogger()
    logger.handlers[0] = RichHandler(markup=True)
    logger = logging.getLogger()
    return logger


test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


train_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_transform(partition: str) -> transforms.Compose:
    if partition == "train":
        return train_transform
    elif partition == "test":
        return test_transform
    elif partition == "val":
        return test_transform
    else:
        raise Exception(f"Preprocessing for partition: {partition} is not implemented")



