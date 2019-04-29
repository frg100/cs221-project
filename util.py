from collections import defaultdict
import os, random, operator, sys
from collections import Counter

def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.
    """
    for f, v in d2.items():
        if (v is not None):
            d1[f] = d1.get(f, 0) + v * scale

def averageVectors(arr):
    """
    Averages all the vectors in arr. Must have same keys
    @param list arr: the list in which the vectors live
    """
    toReturn = {}
    for vec in arr:
        increment(toReturn, 1./len(arr), vec)
    return toReturn


def combineVectors(v1, v2, tag):
    for key in v2.keys():
        newKey = tag + '_' + key
        v1[newKey] = v2.get(key)



def evaluatePredictor(examples, predictor):
    '''
    predictor: a function that takes an x and returns a predicted y.
    Given a list of examples (x, y), makes predictions based on |predict| and returns the fraction
    of misclassiied examples.
    '''
    error = 0
    for index, match in examples.iterrows():
        goalDifference = match['home_team_goal'] - match['away_team_goal']
        y = 1 if goalDifference > 0 else -1
        if predictor(match) != y:
            error += 1
    return 1.0 * error / len(examples)
