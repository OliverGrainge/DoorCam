import time
from pathlib import Path

import data
import mlflow
import models
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from ray.air.integrations.mlflow import MLflowLoggerCallback
from pytorch_metric_learning import losses, miners
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Tuple
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

import mlflow

mlflow.start_run()
mlflow.log_params(config["training"])


def get_optimizer(config: dict, model):
    if config["training"]["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters, lr=config["training"]["learning_rate"])
        return optimizer
    elif config["training"]["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters, lr=config["training"]["learning_rate"])
        return optimizer
    else:
        raise NotImplementedError



class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.FaceIdModule(config)
        self.loss_fn = losses.TripletMarginLoss()
        self.miner = miners.TripletMarginMiner(margin=0.2, trype_of_triplets="all")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x
        
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: torch.Tensor) -> torch.Tensor:
        images, labels, indicies = batch
        embeddings = self.model(images)
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_fn(embeddings, labels, hard_pairs)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: torch.Tensor) -> torch.Tensor:
        images, labels, indicies = batch
        embeddings = self.model(images.to(config["training"]["device"]))
        loss = self.loss_fn(embeddings, labels)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = get_optimizer(config, self.model)
        return optimizer


class DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, mode=None):
        self.train_dataset = data.VGGFaceDataset(config, parition="train")
        self.val_dataset = data.VGGFaceDataset(config, partition="test")

    def train_dataloaders(self) -> DataLoader:
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config["training"]["train_batch_size"],
            shuffle=True,
        )
        return train_dataloader

    def val_dataloaders(self) -> DataLoader:
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=config["training"]["infer_batch_size"],
            shuffle=False,
        )
        return val_dataloader


if __name__ == '__main__':
    tb_logger = pl_loggers.TensorBoardLogger('logs/')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # or another metric that you are logging
        dirpath='model_checkpoints/',
        filename='best_model',
        save_top_k=1,
        mode='min',
    )

    trainer = Trainer(
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=10,
    )
    
    model = LitModel()
    datamodule = DataModule()
    trainer.fit(model, datamodule=datamodule)






