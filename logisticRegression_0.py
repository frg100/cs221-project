import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random

# Define hyperparameters
numFeatures = 86
numOutputs = 3
datasetCSVPath = './basicMatchData.csv'
eta = 1e-5
numEpochs = 100

indicatorMap = {-1: [1, 0, 0], 0: [0, 1, 0], 1: [0, 0, 1]}

# Importing the dataset
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

# Define data loader
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# Define model

model = nn.Linear(numFeatures, 3)
preds = model(inputs)
loss_fn = F.mse_loss
opt = torch.optim.SGD(model.parameters(), lr=eta)

loss = loss_fn(model(inputs), targets)

# Utility function to train model
def fit(num_epochs, model, loss_fn, opt):
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
            print "Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, loss.item())



# Train for numEpochs epochs
fit(numEpochs, model, loss_fn, opt)
preds = model(inputs)
print preds














