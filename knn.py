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
numOutputs = 3
datasetCSVPath = './cleanedBasicMatchData.csv'
eta = 1e-5
numEpochs = 1000
validation_split = .2
shuffle_dataset = True
random_seed= 42
batch_size = 1
k = 1

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
    validation_dl = DataLoader(train_ds, batch_size=1,
                                                    sampler=valid_sampler)

    return inputs, targets, train_dl, validation_dl


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
        if pred != yb:
            errors += 1
        sys.stdout.write("Step [{}/{}], Error: {:.4f}\r".format(step, len(test_dl), errors/(step+1)))
        sys.stdout.flush()
    print ""
    return float(errors)/len(test_dl)


def main():
    print "Loading data..."
    inputs, targets, train_dl, validation_dl = importDataset(datasetCSVPath)
    print "Finished loading data!"


    # Baseline Model
    print "Training model..."
    print knn(train_dl, validation_dl, k)
    print "Finished training model!"


if __name__ == '__main__':
    main()









