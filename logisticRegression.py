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
numFeatures = 122
numOutputs = 3
eta = 0.001
numEpochs = 100
validation_split = .2
shuffle_dataset = True
random_seed= 42
batch_size = 5

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
        if ((epoch + 1) % 1 == 0):
            sys.stdout.write("Epoch [{}/{}], Loss: {:.4f}, Train error: {:.4f}, Test error: {:.4f}\r".format(epoch+1, num_epochs, loss.item(), evaluateModel(evaluate_train_dl, model), evaluateModel(validation_dl, model)))
            sys.stdout.flush()
        #names = 'away_buildUpPlayDribbling,home_buildUpPlayPassing,home_shortPass,home_headers,home_balance,away_finishing,away_defenceDefenderLineClass,away_reactions,home_slidingTackle,home_freeKicks,away_aggression,home_positioning,home_aggression,away_chanceCreationPassing,home_curve,away_longShot,home_gkPositioning,home_sprintSpeed,away_marking,home_finishing,away_vision,home_longPass,WH betting difference,away_headers,home_buildUpPlaySpeed,away_strength,home_acceleration,home_standingTackle,home_marking,away_gkKicking,home_gkHandling,away_curve,home_previous_match_1_result,away_buildUpPlaySpeed,home_dribbling,home_defencePressure,home_gkKicking,home_volleys,home_reactions,IW betting difference,home_defenceTeamWidth,away_gkDiving,home_chanceCreationPassing,away_defenceTeamWidth,home_longShot,home_chanceCreationPositioningClass,home_stamina,away_power,LB betting difference,home_rating,home_previous_match_3_result,home_chanceCreationCrossing,home_agility,VC betting difference,_away_head_to_head,home_defensiveWorkRate,away_agility,away_previous_match_5_result,home_preferredFoot,away_penalties,home_power,home_penalties,away_previous_match_1_result,home_defenceAggression,away_chanceCreationCrossing,home_control,_home_head_to_head,home_previous_match_2_result,home_buildUpPlayPositioningClass,away_balance,home_previous_match_4_result,away_preferredFoot,home_gkReflexes,home_previous_match_5_result,away_rating,away_positioning,B365 betting difference,home_potential,home_crossing,home_defenceDefenderLineClass,BW betting difference,home_interceptions,home_vision,BS betting difference,away_buildUpPlayPassing,home_jump,away_chanceCreationShooting,away_crossing,home_strength,away_shortPass,home_attackingWorkRate,SJ betting difference,GB betting difference,away_acceleration,away_gkHandling,away_gkReflexes,away_jump,home_gkDiving,away_defenceAggression,away_previous_match_3_result,away_standingTackle,away_longPass,away_interceptions,home_chanceCreationShooting,away_control,away_defencePressure,away_chanceCreationPositioningClass,away_previous_match_4_result,away_stamina,away_freeKicks,away_gkPositioning,away_volleys,away_slidingTackle,PS betting difference,away_sprintSpeed,away_buildUpPlayPositioningClass,away_potential,home_buildUpPlayDribbling,away_dribbling,away_previous_match_2_result,away_defensiveWorkRate,away_attackingWorkRate,result'.split(",")
        #print model.weight
        #for x in ["{}: {}".format(names[i], abs(model.weight[i] * 100)) for i in range(len(names))]:
        #   print x
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
    inputs, targets, train_dl, validation_dl, evaluate_train_dl, test_dl = importDataset(datasetCSVPath)
    print "Finished loading data!"

    # Baseline Model
    print "Training model..."
    model = nn.Linear(numFeatures, 3)
    preds = model(inputs)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=eta)
    loss = loss_fn(model(inputs), targets)

    # Train for numEpochs epochs
    fit(numEpochs, model, loss_fn, opt, train_dl, validation_dl, evaluate_train_dl)
    preds = model(inputs)

    trained_weights = model.weight
    print "Finished training model!"

    print
    testError = evaluateModel(test_dl, model)
    print "Final error: {}".format(testError)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: python logisticRegression.py <DATA CSV FILE>"
    else:
        datasetCSVPath = sys.argv[1]
        print "Reading data from {}".format(datasetCSVPath)
        main()









