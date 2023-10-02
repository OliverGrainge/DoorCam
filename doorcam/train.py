import data
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

import doorcam.models as models

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


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
loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

train_triplet_dataset = data.TripletDataset(config, partition="train")
test_triplet_dataset = data.TripletDataset(config, partition="test")


for epoch in range(config["max_epochs"]):
    if config["training"]["mining"] == "partial":
        train_triplet_dataset.partial_mine(model)
    elif config["training"]["mining"] == "random":
        train_triplet_dataset.random_mine(model)
    else:
        raise NotImplementedError

    dataloader = DataLoader(train_triplet_dataset, batch_size=config["training"]["train_batch_size"], shuffle=True)
