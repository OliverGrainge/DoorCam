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


def get_image_paths(config: dict, progressbar: bool = True, partition: Literal["train", "test"] = "train") -> Tuple[np.ndarray, np.ndarray]:
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

    for idx, id in tqdm(enumerate(ids_paths), total=total_paths, disable=not progressbar, desc="Loading Dataset"):
        img_paths = np.array(glob(id + "/*.jpg"))
        ids = np.full(img_paths.shape[0], idx, dtype=int)
        all_img_paths.append(img_paths)
        all_img_ids.append(ids)
        assert len(ids) == len(img_paths)

    all_img_paths = np.hstack(all_img_paths)
    all_img_ids = np.hstack(all_img_ids)

    return all_img_ids, all_img_paths


class ImageDataset(Dataset):
    """
    A custom dataset class for loading images from specified file paths.

    This class extends `torch.utils.data.Dataset` and overrides the `__len__` and
    `__getitem__` methods for custom functionality. It loads images from the specified
    file paths, and applies a given set of transformations to the images (if provided).

    Parameters
    ----------
    image_paths : np.ndarray
        An array of file paths to the images to be loaded.
    transform : Optional[Union[Compose, callable]], optional
        A torchvision.transforms.Compose object or a callable that applies
        transformations to a PIL Image object. Default is None, in which case no
        transformations are applied.

    Attributes
    ----------
    image_paths : np.ndarray
        An array of file paths to the images.
    transform : Optional[Union[Compose, callable]]
        The transformations to be applied to the images.
    """

    def __init__(self, image_paths: np.ndarray, transform: Optional[Union[Compose, callable]] = None) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        """Returns the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Union[np.ndarray, Any]:
        """
        Loads and returns the image at the specified index, applying the specified
        transformations (if any).

        Parameters
        ----------
        idx : int
            The index of the image to be loaded.

        Returns
        -------
        Union[np.ndarray, Any]
            The loaded image, with transformations applied (if any).
        """
        img_path = self.image_paths[idx]
        pil_img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(pil_img)
            return img
        else:
            img = np.array(pil_img)
            return img


class TripletDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create a
    triplet dataset. This dataset returns triplets of images: an anchor image,
    a positive image, and a negative image. The positive images belong to the
    same class as the anchor images, while the negative images belong to a
    different class.

    Parameters
    ----------
    config : dict
        A configuration dictionary containing information for data processing,
        such as the path to the data, batch size, and any other necessary
        information.
    progressbar : bool, optional
        Whether to display a progress bar when loading the data. Default is True.
    partition : {"train", "test"}, optional
        The partition of the data to use. Default is "train".

    Attributes
    ----------
    image_ids : ndarray
        An array containing the ids of all images in the dataset.
    image_paths : ndarray
        An array containing the paths to all images in the dataset.
    config : dict
        The configuration dictionary passed to the constructor.
    partition : {"train", "test"}
        The partition of the data to use.
    index : NoneType
        A placeholder for an index to be used in the `partial_mine` method.
    positives : list
        A list of positive image paths for each anchor image.
    negatives : list
        A list of negative image paths for each anchor image.
    anchors : list
        A list of anchor image paths.

    Methods
    -------
    __len__():
        Returns the number of anchor images in the dataset.
    __getitem__(idx: int):
        Returns a batch of images given an index.
    random_mine(model: nn.Module) -> None:
        Generates triplets using random mining.
    partial_mine(model: nn.Module) -> None:
        Generates triplets using partial mining, which utilizes a pretrained
        model to find hard positives and negatives.
    """

    def __init__(self, config: dict, progressbar: bool = True, partition: Literal["train", "test"] = "train"):
        super().__init__()
        self.image_ids, self.image_paths = get_image_paths(config, progressbar=progressbar, partition=partition)
        self.config = config
        self.partition = partition
        self.index = None
        self.positives = []
        self.negatives = []
        self.anchors = []

    def __len__(self):
        """
        Returns the total number of triplets in the dataset.

        Returns
        -------
        int
            The total number of triplets.
        """
        return len(self.anchors)

    def __getitem__(self, idx):
        """
        Returns a triplet of images for a given index. The batch consists of
        an anchor image, positive images, and negative images.

        Parameters
        ----------
        idx : int
            The index to retrieve the batch of images for.

        Returns
        -------
        torch.Tensor
            A tensor containing the triplet of images. The tensor has the
            following shape: (1 + n_positives + n_negatives, C, H, W), where
                - n_positives is the number of images that are a match to the anchor
                  (specified in the config file),
                - n_negatives is the number of images that do not match the anchor
                  (also specified in the config file),
                - C is the number of channels each image has,
                - H and W are the height and width of the image,
                  as specified in the config file.

        """
        pos_paths = self.positives[idx]
        neg_paths = self.negatives[idx]
        anchor = self.anchors[idx]

        pos_images = [Image.open(pth) for pth in pos_paths]
        neg_images = [Image.open(pth) for pth in neg_paths]
        anchor_images = [Image.open(pth) for pth in anchor]

        all_images = anchor_images + pos_images + neg_images
        if self.partition == "train":
            images = torch.stack([train_transforms(img) for img in all_images])

        elif self.partition == "test":
            images = torch.stack([test_transforms(img) for img in all_images])

        return images

    def random_mine(self, model: nn.Module) -> None:
        """
        Randomly generates triplets of anchor, positive, and negative images.
        The triplets are stored in the `positives`, `negatives`, and `anchors`
        attributes of the dataset object.

        Parameters
        ----------
        model : torch.nn.Module
            The model to use for mining. This parameter is not used in random mining,
            but is kept for compatibility with the `partial_mine` method.
        """

        self.positives, self.negatives, self.anchors = None, None, None
        count = np.arange(len(self.image_ids))
        sample_idx = np.random.choice(count, size=config["training"]["triplet_sample_size"])
        sample_ids = self.image_ids[sample_idx]
        sample_image_paths = self.image_paths[sample_idx]

        for idx, id in tqdm(enumerate(sample_ids), total=len(sample_ids)):
            neg_mask = sample_ids != id
            neg_images = sample_image_paths[neg_mask]
            neg_sample = np.random.choice(neg_images, size=self.config["training"]["n_negatives"])
            self.negatives.append(neg_sample)

            pos_mask = sample_ids == id
            pos_images = sample_image_paths[pos_mask]
            pos_sample = np.random.choice(pos_images, size=self.config["training"]["n_negatives"])
            self.positives.append(pos_sample)

            self.anchors.append(np.array([sample_image_paths[idx]]))

        self.anchors = np.vstack(self.anchors)
        self.positives = np.vstack(self.positives)
        self.negatives = np.vstack(self.negatives)

    def partial_mine(self, model: nn.Module) -> None:
        """
        Generates triplets of anchor, positive, and negative images using partial
        mining. This method uses a pretrained model to find hard positives and
        negatives. The triplets are stored in the `positives`, `negatives`, and
        `anchors` attributes of the dataset object.

        Parameters
        ----------
        model : torch.nn.Module
            The model to use for mining. This should be a pretrained model capable
            of generating embeddings for images.
        """

        self.positives, self.negatives, self.anchors = [], [], []

        model.eval()
        count = np.arange(len(self.image_ids))
        sample_idx = np.random.choice(count, size=config["training"]["triplet_sample_size"])
        sample_ids = self.image_ids[sample_idx]
        sample_image_paths = self.image_paths[sample_idx]

        ds = ImageDataset(sample_image_paths, transform=test_transforms)
        dl = DataLoader(ds, batch_size=self.config["training"]["infer_batch_size"], shuffle=False)

        embeddings = []
        for idx, batch in tqdm(enumerate(dl), desc="Computing Embeddings for Hard Triplet Mining", total=len(dl)):
            desc = model(batch.to(self.config["training"]["device"])).detach().cpu().numpy()
            embeddings.append(desc)
        embeddings = np.vstack(embeddings).astype(np.float32)

        self.index = faiss.IndexFlatL2(self.config["training"]["fc_output_dim"])
        faiss.normalize_L2(embeddings)
        distances, _ = self.index.search(embeddings, k=self.config["training"]["triplet_sample_size"])

        for idx, label in enumerate(sample_ids):
            pos_idx = np.where(sample_ids == label)[0]
            neg_idx = np.where(sample_ids != label)[0]
            pos_idx = pos_idx[pos_idx != idx]

            if len(pos_idx) < self.config["training"]["n_positives"] or len(neg_idx) < self.config["training"]["n_negatives"]:
                continue  # Skip if no available positive or negative examples

            pos_idx = pos_idx[np.argsort(distances[idx, pos_idx])[: self.config["training"]["n_positives"]]]
            neg_idx = neg_idx[np.argsort(-distances[idx, neg_idx])[: self.config["training"]["n_negatives"]]]

            self.negatives.append(sample_image_paths[neg_idx])
            self.positives.append(sample_image_paths[pos_idx])
            self.anchors.append(np.array([sample_image_paths[idx]]))

        self.anchors = np.vstack(self.anchors)
        self.positives = np.vstack(self.positives)
        self.negatives = np.vstack(self.negatives)

        del embeddings
        del self.index
        del distances
        del ds
        del dl


if __name__ == "__main__":
    ds = TripletDataset(config)
    ds.random_mine(4)

    dataloader = DataLoader(ds, batch_size=config["training"]["train_batch_size"])

    for batch in dataloader:
        print(batch.shape)