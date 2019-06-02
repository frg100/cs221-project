import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import sys
import matplotlib.pyplot as plt


target_names = ['home win', 'tie', 'home lose']
indicatorMap = {-1: 0, 0: 1, 1: 2}


def PCA(data, k=2):
    # preprocess the data
    X = torch.from_numpy(data)
    X_mean = torch.mean(X,0)
    X = X - X_mean.expand_as(X)

    # svd
    U,S,V = torch.svd(torch.t(X))
    return torch.mm(X,U[:,:k])


def main():
    # Importing the dataset
    df = pd.read_csv(datasetCSVPath)
    df = np.array(df.values, dtype='float32')
    df = df[~np.isnan(df).any(axis=1)]
    # Separate into inputs and targets by last column
    X = np.array(df[:, :-1], dtype='float32')
    y = np.array([indicatorMap[int(y)] for y in df[:, -1]], dtype='int64')


    X_PCA = PCA(X)

    plt.figure()

    for i, target_name in enumerate(target_names):
        plt.scatter(X_PCA[y == i, 0], X_PCA[y == i, 1], label=target_name)

    plt.legend()
    plt.title('PCA of Match dataset')
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: python pca.py <DATA CSV FILE>"
    else:
        datasetCSVPath = sys.argv[1]
        print "Reading data from {}".format(datasetCSVPath)
        main()

