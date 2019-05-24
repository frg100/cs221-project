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
n_in, n_h, n_out = 10, 10, 3
#n_in, n_h, n_out = 10, 10, 3
eta = 0.001
numEpochs = 50
validation_split = .2
shuffle_dataset = True
random_seed= 42
batch_size = 100

indicatorMap = {-1: 0, 0: 1, 1: 2}


# Importing the dataset
def importDataset(datasetCSVPath):
    testCSVPath = "test_" + datasetCSVPath
    trainCSVPath = "train_" + datasetCSVPath

    df = pd.read_csv(trainCSVPath)
    df = np.array(df.values, dtype='float32')
    testDF = pd.read_csv(testCSVPath)
    testDF = np.array(testDF.values, dtype='float32')

    df = df[~np.isnan(df).any(axis=1)]
    testDF = testDF[~np.isnan(testDF).any(axis=1)]

    # Separate into inputs and targets by last column
    inputs = np.array(df[:, :-1], dtype='float32')
    targets = np.array([indicatorMap[int(y)] for y in df[:, -1]], dtype='int64')
    test_inputs = np.array(testDF[:, :-1], dtype='float32')
    test_targets = np.array([indicatorMap[int(y)] for y in testDF[:, -1]], dtype='int64')

    # Turn into tensors
    inputs = torch.from_numpy(inputs)
    targets = torch.from_numpy(targets)
    test_inputs = torch.from_numpy(test_inputs)
    test_targets = torch.from_numpy(test_targets)

    # Define dataset
    train_ds = TensorDataset(inputs, targets)
    test_ds = TensorDataset(test_inputs, test_targets)

    # Creating data indices for training and validation splits:
    dataset_size = len(inputs)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
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
    test_dl = DataLoader(test_ds, batch_size=1)

    return inputs, targets, train_dl, validation_dl, evaluate_train_dl, test_dl


# Stochastic Gradient Descent
def fit(num_epochs, model, loss_fn, opt, train_dl, validation_dl, evaluate_train_dl):
    for epoch in range(num_epochs):
        # Train in batches
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            # Perform backpropagation
            loss.backward()
            # Update parameters using gradient
            opt.step()
            # Reset the gradient to 0
            opt.zero_grad()

        # Print progress
        if ((epoch + 1) % 1 == 0):
            sys.stdout.write("Epoch [{}/{}], Loss: {:.4f}, Train error: {:.4f}, Dev error: {:.4f}\r".format(epoch+1, num_epochs, loss.item(), evaluateModel(evaluate_train_dl, model), evaluateModel(validation_dl, model)))
            sys.stdout.flush()
        # print model[0].weight.data
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


def main():
    print "Loading data..."
    inputs, targets, train_dl, validation_dl, evaluate_train_dl, test_dl = importDataset(datasetCSVPath)
    print "Finished loading data!"

    # BModel
    print "Training model..."
    # Defines a neural network with a ReLU hidden layer and a sigmoid output layer
    model = nn.Sequential(nn.Linear(n_in, n_h),
                     nn.ReLU(),
                     nn.Linear(n_h, n_out),
                     nn.Sigmoid())
    preds = model(inputs)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=eta)
    loss = loss_fn(model(inputs), targets)

    # Train for numEpochs epochs
    fit(numEpochs, model, loss_fn, opt, train_dl, validation_dl, evaluate_train_dl)

    print "Finished training model!"

    print
    testError = evaluateModel(test_dl, model)
    print "Final error: {}".format(testError)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: python neuralNetwork.py <DATA CSV FILE>"
    else:
        datasetCSVPath = sys.argv[1]
        print "Reading data from {}".format(datasetCSVPath)
        main()









