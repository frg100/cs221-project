import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import sys

# Define hyperparameters
numFeatures = 86
oracleNumFeatures = {'': 90, 'Short': 4, 'Goals': 1}
numOutputs = 3
datasetCSVPath = './basicMatchData.csv'
oracleDatasetCSVPath = './basicMatchDataOracle.csv'
eta = 1e-5
numEpochs = 1000
validation_split = .2
shuffle_dataset = True
random_seed= 42
batch_size = 5

indicatorMap = {-1: 0, 0: 1, 1: 2}


# Importing the dataset
def importDataset(datasetCSVPath):
    df = pd.read_csv(datasetCSVPath)
    df = np.array(df.values, dtype='float32')
    df = df[~np.isnan(df).any(axis=1)]
    # Separate into inputs and targets by last column
    inputs = np.array(df[:, :-1], dtype='float32')
    targets = np.array([indicatorMap[int(y)] for y in df[:, -1]], dtype='int64')

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
    print "Dataset size: {}, Train DL size: {}, Test DL size: {}".format(dataset_size, len(train_indices), len(val_indices))

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
            sys.stdout.write("Epoch [{}/{}], Loss: {:.4f}, Train error: {:.4f}, Test error: {:.4f}\r".format(epoch+1, num_epochs, loss.item(), evaluateModel(evaluate_train_dl, model), evaluateModel(validation_dl, model)))
            sys.stdout.flush()
    print ""


def evaluateModel(dl, model):
    m =  nn.Softmax(dim=1)
    errors = 0.
    for xb, yb in dl:
        pred = model(xb)
        # Checks whether or not the max value in the arrays are the same => both have the same result predicted
        if (np.argmax(m(pred).detach().numpy()) != yb.numpy()[0]):
            errors += 1
    return float(errors)/len(dl)


def nearestNeighborY(dl, v):
    pdist = nn.PairwiseDistance(p=2)

    bestSimilarity = 0.
    bestY = None
    for step, (xb, yb) in enumerate(dl):
        similarity = pdist(xb, v)
        if (similarity > bestSimilarity):
            bestSimilarity = similarity
            bestY = yb
    return bestY

def knn(train_dl, test_dl, k):
    errors = 0.
    for step, (xb, yb) in enumerate(test_dl):
        pred = nearestNeighborY(train_dl, xb)
        if (np.argmax(pred.detach().numpy(), axis=1) != np.argmax(yb.detach().numpy(), axis=1)):
            errors += 1
        sys.stdout.write("Step [{}/{}], Error: {:.4f}\r".format(step, len(test_dl), errors/(step+1)))
        sys.stdout.flush()
    print ""
    return float(errors)/len(test_dl)


def main():
    print "Loading data..."
    inputs, targets, train_dl, validation_dl, evaluate_train_dl = importDataset(datasetCSVPath)
    print "Finished loading data!"

    # Oracle model
    for edit in ['', 'Goals', 'Short']:
        print "Training {} oracle...".format(edit)
        #knn(train_dl, validation_dl, 1)
        oracle_inputs, oracle_targets, oracle_train_dl, oracle_validation_dl, oracle_evaluate_train_dl = importDataset('./basicMatchDataOracle{}.csv'.format(edit))
        oracle_model = nn.Linear(oracleNumFeatures[edit], 3)
        oracle_preds = oracle_model(oracle_inputs)
        oracle_loss_fn = torch.nn.CrossEntropyLoss()
        oracle_opt = torch.optim.SGD(oracle_model.parameters(), lr=eta)
        #oracle_loss = oracle_loss_fn(oracle_model(oracle_inputs), oracle_targets)

        # Train for numEpochs epochs
        fit(numEpochs, oracle_model, oracle_loss_fn, oracle_opt, oracle_train_dl, oracle_validation_dl, oracle_evaluate_train_dl)
        oracle_preds = oracle_model(oracle_inputs)

        oracle_trained_weights = oracle_model.weight

        #print oracle_preds, oracle_targets, oracle_trained_weights
        print "Finished training {} oracle!".format(edit)

    # Baseline Model
    print "Training baseline..."
    model = nn.Linear(numFeatures, 3)
    preds = model(inputs)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=eta)
    loss = loss_fn(model(inputs), targets)

    # Train for numEpochs epochs
    fit(numEpochs, model, loss_fn, opt, train_dl, validation_dl, evaluate_train_dl)
    preds = model(inputs)

    trained_weights = model.weight
    print "Finished training baseline!"


if __name__ == '__main__':
    main()









