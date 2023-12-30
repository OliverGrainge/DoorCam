import time
from pathlib import Path
from typing import Tuple

import data
import faiss
import mlflow
import models
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_metric_learning import losses, miners, reducers
from pytorch_metric_learning.samplers import MPerClassSampler
from pytorch_metric_learning.utils.accuracy_calculator import \
    AccuracyCalculator
from ray.air.integrations.mlflow import MLflowLoggerCallback
from sklearn.metrics import pairwise_distances_argmin_min
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import get_config, test_transform

config = get_config()


from torchvision import models


class ResNet50Embedding(nn.Module):
    def __init__(self):
        super(ResNet50Embedding, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.embedding = nn.Linear(
            resnet.fc.in_features, 1024
        )  # Example embedding size

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return nn.functional.normalize(x, p=2, dim=1)


class TripletModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ResNet50Embedding()
        self.loss_fn = losses.TripletMarginLoss(
            margin=0.1, swap=False, smooth_loss=False, triplets_per_anchor="all"
        )
        self.miner = miners.TripletMarginMiner(margin=0.1, type_of_triplets="semihard")
        self.feature_dim = self.feature_size()

    def feature_size(self):
        image = torch.randn(1, 3, 224, 224)
        self.model.cpu()
        out = self.model(image)
        return out.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

    def predict(self, x: Image) -> np.ndarray:
        x = test_transform(x)
        features = self.model(x[None, :])
        return features.detach().cpu().numpy().flatten()

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        images, labels, indicies = batch
        embeddings = self.model(images)
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_fn(embeddings, labels, hard_pairs)

        self.log("train_loss", loss)
        return loss

    def on_validation_epoch_start(self) -> None:
        val_dataset = self.trainer.datamodule.val_dataset
        self.val_descriptors = np.empty(
            (len(val_dataset.image_paths), self.feature_dim), dtype=np.float32
        )

    def validation_step(self, batch, batch_idx) -> None:
        images, labels, indicies = batch
        embeddings = self(images).detach().cpu().numpy().astype(np.float32)
        self.val_descriptors[indicies.detach().cpu().numpy(), :] = embeddings

    def on_validation_epoch_end(self) -> None:
        val_dataset = self.trainer.datamodule.val_dataset
        labels = np.array(val_dataset.image_ids)  # Ensure labels are in a NumPy array
        boundary = int(0.2 * self.val_descriptors.shape[0])
        query_desc = self.val_descriptors[boundary:]
        query_labels = labels[boundary:]
        map_desc = self.val_descriptors[:boundary]
        map_labels = labels[:boundary]
        index = faiss.IndexFlatL2(map_desc.shape[1])
        index.add(map_desc)
        k = 3  # Example value for k
        _, top_k_indices = index.search(query_desc, k)
        recall_count = 0
        valid_query_count = 0
        for idx, query_label in enumerate(query_labels):
            if query_label in map_labels:
                valid_query_count += 1
                retrieved_labels = map_labels[top_k_indices[idx]]
                if query_label in retrieved_labels:
                    recall_count += 1
        if valid_query_count > 0:
            recall_at_k = recall_count / valid_query_count
        else:
            recall_at_k = 0

        print(" ")
        print(" ")
        print(f"Recall@{k}", recall_at_k)
        print(" ")
        print(" ")
        self.log(f"Recall@{k}", recall_at_k)

    def configure_optimizers(self) -> optim.Optimizer:
        if config["training"]["optimizer"] == "adam":
            optimizer = optim.Adam(
                self.model.parameters(), lr=config["training"]["learning_rate"]
            )
            return optimizer
        elif config["training"]["optimizer"] == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(), lr=config["training"]["learning_rate"]
            )
            return optimizer
        else:
            raise NotImplementedError


class DataModule(pl.LightningDataModule):
    def __init__(self, sampler=True):
        super().__init__()
        self.sampler = sampler

    def setup(self, stage=None):
        self.train_dataset = data.VGGFaceDataset(
            config, partition="train", num_ids=5000
        )
        self.val_dataset = data.VGGFaceDataset(config, partition="test", num_ids=25)

        print(
            "==================== train dataset has: ",
            self.train_dataset.__len__(),
            " samples",
        )
        print(
            "==================== test dataset has: ",
            self.val_dataset.__len__(),
            " samples",
        )

        if self.sampler:
            self.train_sampler = MPerClassSampler(
                self.train_dataset.image_ids,
                config["training"]["batch_samples_per_class"],
                batch_size=config["training"]["train_batch_size"],
                length_before_new_iter=200000,
            )

    def train_dataloader(self) -> DataLoader:
        if self.sampler:
            train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=config["training"]["train_batch_size"],
                num_workers=config["training"]["num_workers"],
                sampler=self.train_sampler,
            )
        else:
            train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=config["training"]["train_batch_size"],
                num_workers=config["training"]["num_workers"],
            )

        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=config["training"]["infer_batch_size"],
            num_workers=config["training"]["num_workers"],
            drop_last=False,
        )
        return val_dataloader


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    tb_logger = pl_loggers.TensorBoardLogger("logs/")

    checkpoint_callback = ModelCheckpoint(
        monitor="Recall@3",  # or another metric that you are logging
        dirpath="model_checkpoints/",
        filename="best_model",
        save_top_k=1,
        mode="min",
    )

    trainer = Trainer(
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=50,
        precision="16-mixed",
        num_sanity_val_steps=0,
        # limit_train_batches=10
    )

    model = TripletModel()
    datamodule = DataModule(sampler=True)
    trainer.fit(model, datamodule=datamodule)
