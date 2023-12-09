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


model = models.FaceIDModel(config).to(config["training"]["device"])
loss_fn = losses.TripletMarginLoss()
miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="all")
optimizer = get_optimizer(config, model)

train_dataset = data.VGGFaceDataset(config, partition="train")
test_dataset = data.VGGFaceDataset(config, partition="test")

train_dataloader = DataLoader(
    train_dataset,
    batch_size=config["training"]["train_batch_size"],
    shuffle=True,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=config["training"]["infer_batch_size"],
    shuffle=False,
)

test_losses = []
train_losses = []

for epoch in range(config["max_epochs"]):
    # Training Loop
    model.train()
    average_train_loss = []
    for batch, labels, indicies in tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        desc="Training Epoch " + str(epoch),
    ):
        optimizer.zero_grad()
        embeddings = model(embeddings.to(config["training"]["device"]))
        hard_pairs = miner(embeddings, labels)
        train_loss = loss_fn(embeddings, labels, hard_pairs)

        train_loss.backward()
        optimizer.step()
        average_train_loss.append(train_loss)

    # Validation Loop
    model.eval()
    average_test_loss = []
    for batch, labels, idx in tqdm(
        enumerate(test_dataloader),
        total=len(test_dataloader),
        desc="Testing" + str(epoch),
    ):
        with torch.no_grad():
            embeddings = model(batch.to(config["training"]["device"]))
            test_loss = loss_fn(embeddings, labels)
        average_test_loss.append(test_loss)

    print("============ Average Test Loss ", +str(np.mean(average_test_loss)))
    mlflow.log_metrics({"train_loss": np.mean(average_train_loss), "val_loss": np.mean(average_test_loss)}, step=epoch)

mlflow.pytorch.log_model(model, "model")
mlflow.end_run()
