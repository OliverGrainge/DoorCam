import data
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

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
optimizer = get_optimizer(config, model)

train_triplet_dataset = data.TripletDataset(config, partition="train")
test_triplet_dataset = data.TripletDataset(config, partition="test")


test_losses = []

for epoch in range(config["max_epochs"]):
    model.train
    if config["training"]["mining"] == "partial":
        train_triplet_dataset.partial_mine(model)
    elif config["training"]["mining"] == "random":
        train_triplet_dataset.random_mine(model)
    else:
        raise NotImplementedError

    train_dataloader = DataLoader(train_triplet_dataset, batch_size=config["training"]["train_batch_size"], shuffle=True)

    for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training Epoch " + str(epoch)):
        optimizer.zero_grad()
        
        # reshape embedings 
        anchors = batch[:, 0, :, :, :]
        positives = batch[:, 1, :, :, :]
        negatives = batch[:, 2, :, :, :]

        images = torch.vstack((anchors, positives, negatives))

        embeddings = model(images.to(config["training"]["device"]))

        # reshape embedings 

        anchor_embeddings = embeddings[:config["training"]["train_batch_size"]]
        positive_embeddings = embeddings[config["training"]["train_batch_size"]:config["training"]["train_batch_size"]*2]
        negative_embeddings = embeddings[config["training"]["train_batch_size"]*2:]

        loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
        loss.backward()
        optimizer.step()

    
    del train_dataloader
    if config["training"]["mining"] == "partial":
        test_triplet_dataset.partial_mine(model)
    elif config["training"]["mining"] == "random":
        test_triplet_dataset.random_mine(model)
    else:
        raise NotImplementedError

    test_dataloader = DataLoader(test_triplet_dataset, batch_size=config["training"]["infer_batch_size"], shuffle=False)
    model.eval()
    
    average_test_loss = []
    for idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Testing" + str(epoch)):
        # reshape embedings 
        anchors = batch[:, 0, :, :, :]
        positives = batch[:, 1, :, :, :]
        negatives = batch[:, 2, :, :, :]

        images = torch.vstack((anchors, positives, negatives))
        with torch.no_grad()
            embeddings = model(images.to(config["training"]["device"]))

        # reshape embedings 

        anchor_embeddings = embeddings[:config["training"]["train_batch_size"]]
        positive_embeddings = embeddings[config["training"]["train_batch_size"]:config["training"]["train_batch_size"]*2]
        negative_embeddings = embeddings[config["training"]["train_batch_size"]*2:]

        loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings).detach().cpu().numpy()
        average_test_loss.append(loss)

    print("============ Average Test Loss ", + str(np.mean(average_test_loss)))






        
