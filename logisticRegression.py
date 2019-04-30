import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random

# Define hyperparameters
numFeatures = 86
numOutputs = 3
datasetCSVPath = './basicMatchData.csv'
eta = 1e-5
numEpochs = 10000
validation_split = .2
shuffle_dataset = True
random_seed= 42
batch_size = 5

indicatorMap = {-1: [1, 0, 0], 0: [0, 1, 0], 1: [0, 0, 1]}

# Importing the dataset
def importDataset(datasetCSVPath):
    df = pd.read_csv(datasetCSVPath)
    df = np.array(df.values, dtype='float32')
    df = df[~np.isnan(df).any(axis=1)]
    # Separate into inputs and targets by last column
    inputs = np.array(df[:, :-1], dtype='float32')
    targets = np.array([indicatorMap[int(y)] for y in df[:, -1]], dtype='float32')

    # Turn into tensors
    inputs = torch.from_numpy(inputs)
    targets = torch.from_numpy(targets)

    # Define dataset
    train_ds = TensorDataset(inputs, targets)

    # Creating data indices for training and validation splits:
    dataset_size = len(inputs)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    print dataset_size, len(train_indices), len(val_indices)

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_dl = DataLoader(train_ds, batch_size=batch_size, 
                                               sampler=train_sampler)
    evaluate_train_dl = DataLoader(train_ds, batch_size=1, 
                                               sampler=train_sampler)
    validation_dl = DataLoader(train_ds, batch_size=1,
                                                    sampler=valid_sampler)

    return inputs, targets, train_dl, validation_dl, evaluate_train_dl


# Utility function to train model
def fit(num_epochs, model, loss_fn, opt, train_dl, validation_dl, evaluate_train_dl):
    for epoch in range(num_epochs):
        # Train in batches
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            # Calculate gradient
            loss.backward()
            # Update parameters using gradient
            opt.step()
            # Reset the gradient to 0
            opt.zero_grad()

        # Print progress
        if ((epoch + 1) % 10 == 0):
            print "Epoch [{}/{}], Loss: {:.4f}, Train error: {:.4f}, Test error: {:.4f}".format(epoch+1, num_epochs, loss.item(), evaluateModel(evaluate_train_dl, model), evaluateModel(validation_dl, model))


def evaluateModel(dl, model):
    errors = 0.
    for xb, yb in dl:
        pred = model(xb)
        if (np.argmax(pred.detach().numpy(), axis=1) != np.argmax(yb.detach().numpy(), axis=1)):
            errors += 1
    return float(errors)/len(dl)


def main():
    inputs, targets, train_dl, validation_dl, evaluate_train_dl = importDataset(datasetCSVPath)

    # Define model
    model = nn.Linear(numFeatures, 3)
    preds = model(inputs)
    loss_fn = F.mse_loss
    opt = torch.optim.SGD(model.parameters(), lr=eta)
    loss = loss_fn(model(inputs), targets)

    # Train for numEpochs epochs
    fit(numEpochs, model, loss_fn, opt, train_dl, validation_dl, evaluate_train_dl)
    preds = model(inputs)

    print model, model.parameters(), model.weight
    trained_weights = model.weight
    print model, trained_weights



if __name__ == '__main__':
    main()









