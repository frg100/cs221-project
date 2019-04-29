#import csv
import numpy as np
import pandas as pd
import sqlite3
from collections import defaultdict
from util import *


# Table names
# Country, League, Match, Player, Player_Attributes, Team, Team_Attributes
conn = sqlite3.connect("database.sqlite")


def readData():
    """
    Read in match, player, and player attribute data
    """
    print "Reading in match data..."
    matches = pd.read_sql_query("select * from Match;", conn)
    print "Reading in player data..."
    players = pd.read_sql_query("select * from Player;", conn)
    print "Reading in player attribute data..."
    playerAttributes = pd.read_sql_query("select * from Player_Attributes;", conn)
    print "Finished reading in data!"

    return matches, players, playerAttributes



def getPlayerAttributes(playerId, season):
    if (np.isnan(playerId)):
        return {}
    earlyDate = '{}-07-01 00:00:00'.format(season.split("/")[0])
    lateDate = '{}-07-01 00:00:00'.format(season.split("/")[1])
    query = "SELECT * FROM 'Player_Attributes' WHERE player_api_id IS {} AND date > '{}' AND date < '{}' limit 1".format(playerId, earlyDate, lateDate)
    players = pd.read_sql_query(query, conn)

    workRateToInt = defaultdict(lambda: 1)
    workRateToInt['low'] = 0
    workRateToInt['medium'] = 1
    workRateToInt['high'] = 2

    attributes = defaultdict(float)
    for index, player in players.iterrows():
        attributes['rating'] = player['overall_rating']
        attributes['potential'] = player['potential']
        attributes['preferredFoot'] = 0 if player['preferred_foot'] == 'left' else 1
        attributes['attackingWorkRate'] = workRateToInt[player['attacking_work_rate']]
        attributes['defensiveWorkRate'] = workRateToInt[player['defensive_work_rate']]
        attributes['crossing'] = player['crossing']    
        attributes['finishing'] = player['finishing']   
        attributes['headers'] = player['heading_accuracy']    
        attributes['shortPass'] = player['short_passing']   
        attributes['volleys'] = player['volleys']   
        attributes['dribbling'] = player['dribbling']
        attributes['curve'] = player['curve']   
        attributes['freeKicks'] = player['free_kick_accuracy']  
        attributes['longPass'] = player['long_passing']    
        attributes['control'] = player['ball_control']    
        attributes['acceleration'] = player['acceleration']    
        attributes['sprintSpeed'] = player['sprint_speed']   
        attributes['agility'] = player['agility']   
        attributes['reactions'] = player['reactions']   
        attributes['balance'] = player['balance'] 
        attributes['power'] = player['shot_power']  
        attributes['jump'] = player['jumping'] 
        attributes['stamina'] = player['stamina'] 
        attributes['strength'] = player['strength']    
        attributes['longShot'] = player['long_shots']  
        attributes['aggression'] = player['aggression']  
        attributes['interceptions'] = player['interceptions']   
        attributes['positioning'] = player['positioning'] 
        attributes['vision'] = player['vision']  
        attributes['penalties'] = player['penalties']   
        attributes['marking'] = player['marking'] 
        attributes['standingTackle'] = player['standing_tackle'] 
        attributes['slidingTackle'] = player['sliding_tackle']  
        attributes['gkDiving'] = player['gk_diving']   
        attributes['gkHandling'] = player['gk_handling'] 
        attributes['gkKicking'] = player['gk_kicking']  
        attributes['gkPositioning'] = player['gk_positioning']  
        attributes['gkReflexes'] = player['gk_reflexes']
    return attributes



def calculatePlayerAttributeFeatures(match, phi, season):
    homePlayers = [match['home_player_%d' %i] for i in range(1, 12)]
    awayPlayers = [match['away_player_%d' %i] for i in range(1, 12)]

    homePlayerAttributes = [getPlayerAttributes(playerId, season) for playerId in homePlayers]
    awayPlayerAttributes = [getPlayerAttributes(playerId, season) for playerId in awayPlayers]

    homeVector = averageVectors(homePlayerAttributes)
    awayVector = averageVectors(awayPlayerAttributes)

    # TODO: Average by player type (gk fields should only be calculated for the gk playing)

    combineVectors(phi, homeVector, 'home')
    combineVectors(phi, awayVector, 'away')


def calculateBettingFeatures(match, phi):
    bettingHouses = ['B365', "BW", "IW", "LB", "PS", "WH", "SJ", "VC", "GB", "BS"]
    for house in bettingHouses:
        homePercent = 1./match[house + "H"]
        awayPercent = 1./match[house + "A"]
        phi["{} betting difference".format(house)] = homePercent - awayPercent


def extractFeatures(match):
    phi = defaultdict(float)

    season = match['season']

    calculatePlayerAttributeFeatures(match, phi, season)
    calculateBettingFeatures(match, phi)

    return phi

def main(matches, players, playerAttributes):
    trainExamples =  matches.sample(100)
    testExamples = matches.sample(20)
    print "Training data to learn weights"
    weights = learnPredictor(trainExamples, testExamples, extractFeatures, numIters=20, eta=0.01)
    print weights


def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    predictor = lambda x : 1 if dotProduct(featureExtractor(x), weights) >= 0 else -1
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    for t in range(numIters):
        print "Starting iteration: " + str(t)
        for index, match in trainExamples.iterrows():
            print weights
            goalDifference = match['home_team_goal'] - match['away_team_goal']
            y = 1 if goalDifference > 0 else -1
            phi = featureExtractor(match)
            margin = dotProduct(weights, phi)*y
            gradient = {}
            if margin < 1:
                increment(gradient, (-1 * y), phi)
            increment(weights, -1 * eta, gradient)
        print "Train error: %f, Test error: %f" %(evaluatePredictor(trainExamples, predictor), evaluatePredictor(testExamples, predictor))
    # END_YOUR_CODE
    return weights


if __name__ == '__main__':
    matches, players, playerAttributes = readData()
    main(matches, players, playerAttributes)