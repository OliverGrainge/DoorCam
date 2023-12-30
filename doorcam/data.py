from glob import glob
from typing import Any, Dict, Literal, Optional, Tuple, Union

import faiss
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose
from tqdm import tqdm
from utils import get_config, get_transform, get_logger

config = get_config()
logger = get_logger()


def reduce_dataset(image_paths, labels, samples_per_category):
    unique_labels = np.unique(labels)
    reduced_image_paths = []
    reduced_labels = []

    for label in unique_labels:
        # Find indices where current label occurs
        indices = np.where(labels == label)[0]

        # If there are fewer samples than desired, keep them all
        if len(indices) <= samples_per_category:
            reduced_image_paths.extend(image_paths[indices])
            reduced_labels.extend(labels[indices])
        else:
            # Randomly select desired number of samples
            selected_indices = np.random.choice(
                indices, samples_per_category, replace=False
            )
            reduced_image_paths.extend(image_paths[selected_indices])
            reduced_labels.extend(labels[selected_indices])
    return np.array(reduced_image_paths).flatten(), np.array(reduced_labels).flatten()


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
    logger.info(f"Total number of image paths in VGG {total_paths}")
    logger.info(f"Total number of ID's in VGG {ids_paths}")
    assert len(total_paths) == len(ids_paths)

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
    indicies = np.random.shuffle(indices)
    all_img_paths[:] = all_img_paths[indicies]
    all_img_ids[:] = all_img_ids[indicies]
    return all_img_ids, all_img_paths


class VGGFaceDataset(Dataset):
    def __init__(
        self, config: dict, partition: str, num_ids: int = 500, debug: bool = False
    ) -> None:
        self.transform = get_transform(partition)
        self.image_ids, self.image_paths = get_image_paths(config, partition=partition)
        if debug:
            ids = np.arange(20)
            indicies = []
            for number in ids:
                idx = np.where(self.image_ids == number)
                indicies.extend(idx[0])

            self.image_ids = self.image_ids[indicies]
            self.image_paths = self.image_paths[indicies]

            indices = np.arange(len(self.image_ids))
            np.random.shuffle(indices)
            self.image_ids[:] = self.image_ids[indices]
            self.image_paths[:] = self.image_paths[indices]

            self.image_ids = self.image_ids[:2000]
            self.image_paths = self.image_paths[:2000]
            assert len(self.image_ids) == len(self.image_paths)

        ids = np.arange(num_ids)
        indicies = []
        for number in ids:
            idx = np.where(self.image_ids == number)
            indicies.extend(idx[0])

        self.image_ids = self.image_ids[indicies]
        self.image_paths = self.image_paths[indicies]

        indices = np.arange(len(self.image_ids))
        np.random.shuffle(indices)
        self.image_ids[:] = self.image_ids[indices]
        self.image_paths[:] = self.image_paths[indices]

        if partition in ["val", "test"]:
            self.image_paths, self.image_ids = reduce_dataset(
                self.image_paths, self.image_ids, 100
            )
            indicies = np.arange(len(self.image_paths))
            np.random.shuffle(indicies)
            self.image_paths = self.image_paths[indicies]
            self.image_ids = self.image_ids[indicies]

        assert len(self.image_ids) == len(self.image_ids)

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
