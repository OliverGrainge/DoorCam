from glob import glob
from typing import Any, Literal, Optional, Tuple, Union

import faiss
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose
from tqdm import tqdm
from utils import test_transforms, train_transforms

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


def get_image_paths(
    config: dict,
    progressbar: bool = True,
    partition: Literal["train", "test"] = "train",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect paths of training images and their respective IDs based on a given directory structure.

    This function iterates through the directory specified by `config["training"]["vggface_dir"]`
    to collect paths of all training imagpartition: ):es. It also maintains a corresponding ID for each image
    based on the sub-directory they are located in.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing the path of training data.
        - config["training"]["vggface_dir"] (str): Path to the directory containing training data.
    progressbar : bool, optional
        Flag to enable or disable the progress bar while processing.
        Default is True.
    partition : {"train", "test"}, optional
        The partition of the data to use. Default is "train".

    -------
    tuple
        A tuple containing two elements:
        - all_img_ids (np.ndarray): An array of IDs corresponding to the images.
        - all_img_paths (np.ndarray): An array of paths to the training images.
    """
    vggface_dir = config["training"]["vggface_dir"] + "/data/" + partition + "/*"
    ids_paths = glob(vggface_dir)
    total_paths = len(ids_paths)

    all_img_paths = []
    all_img_ids = []

    for idx, id in tqdm(
        enumerate(ids_paths),
        total=total_paths,
        disable=not progressbar,
        desc="Loading Dataset",
    ):
        img_paths = np.array(glob(id + "/*.jpg"))
        ids = np.full(img_paths.shape[0], idx, dtype=int)
        all_img_paths.append(img_paths)
        all_img_ids.append(ids)
        assert len(ids) == len(img_paths)

    all_img_paths = np.hstack(all_img_paths)
    all_img_ids = np.hstack(all_img_ids)
    indices = np.arange(len(all_img_ids))
    np.random.shuffle(indices)
    all_img_paths[:] = all_img_paths[indices]
    all_img_ids[:] = all_img_ids[indices]
    return all_img_ids, all_img_paths


class VGGFaceDataset(Dataset):
    def __init__(
        self,
        image_paths: np.ndarray,
        transform: Optional[Union[Compose, callable]] = None,
    ) -> None:
        self.image_paths = image_paths
        self.transform = transform
        self.image_ids, self.image_paths = get_image_paths()

    def __len__(self) -> int:
        """Returns the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Union[np.ndarray, Any]:
        img_path = self.image_paths[idx]
        img_id = self.image_ids[idx]
        pil_img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(pil_img)
            return img, img_id, idx
        else:
            img = np.array(pil_img)
            return img, img_id, idx
