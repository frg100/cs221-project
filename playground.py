#import csv
import numpy as np
import pandas as pd
import sqlite3
from collections import defaultdict
from util import *
import csv
import time
import datetime
import re

fileNameToWriteTo = 'basicMatchDataOracleGoals.csv'
print "Writing to {}".format(fileNameToWriteTo)


# Table names
# Country, League, Match, Player, Player_Attributes, Team, Team_Attributes
conn = sqlite3.connect("database.sqlite")


def readData():
    """
    Read in match, player, and player attribute data
    """
    print "Reading in match data..."
    matches = pd.read_sql_query("select * from Match WHERE possession IS NOT null AND shoton IS NOT null;", conn)
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

def calculatePossession(match, phi):
    possession = match['possession']
    pattern = '<elapsed>90</elapsed>.*<awaypos>(.*)</awaypos><homepos>(.*)</homepos>.*</possession>'
    a = re.search(pattern, possession)
    if (a):
        homePos = a.group(1)
        awayPos = a.group(2)

        phi['home_possession'] = homePos
        phi['away_possession'] = awayPos


def calculateShotsOnGoal(homeTeam, awayTeam, match, phi):
    shotsOn = match['shoton']
    pattern = '<team>([0-9]*)</team>'
    groups = re.findall(pattern, shotsOn)
    if(groups):
        phi['home_shots_on_goal'] = 0.
        phi['away_shots_on_goal'] = 0.
        for group in groups:
            if (int(group) == homeTeam):
                phi['home_shots_on_goal'] += 1
            elif (int(group) == awayTeam):
                phi['away_shots_on_goal'] += 1




def extractFeatures(match):
    phi = defaultdict(float)

    season = match['season']
    homeTeam = match['home_team_api_id']
    awayTeam = match['away_team_api_id']

    #calculatePlayerAttributeFeatures(match, phi, season)
    #calculateBettingFeatures(match, phi)
    #calculatePossession(match, phi)
    #calculateShotsOnGoal(homeTeam, awayTeam, match, phi)
    phi['goal_difference'] = match['home_team_goal'] - match['away_team_goal']

    return phi


    

def main(matches, players, playerAttributes):
    with open(fileNameToWriteTo, mode='w') as csv_file:
        #fieldnames = ['away_possession', 'home_possession', 'home_shots_on_goal', 'away_shots_on_goal', 'home_shortPass', 'home_headers', 'home_balance', 'away_finishing', 'away_reactions', 'home_slidingTackle', 'home_freeKicks', 'away_aggression', 'home_positioning', 'home_aggression', 'home_curve', 'away_longShot', 'home_gkPositioning', 'home_sprintSpeed', 'away_marking', 'home_finishing', 'away_vision', 'home_longPass', 'WH betting difference', 'away_headers', 'away_strength', 'home_acceleration', 'home_standingTackle', 'home_marking', 'away_gkKicking', 'home_gkHandling', 'away_curve', 'home_dribbling', 'home_gkKicking', 'home_volleys', 'home_reactions', 'IW betting difference', 'away_gkDiving', 'home_longShot', 'home_stamina', 'away_power', 'LB betting difference', 'home_rating', 'home_agility', 'VC betting difference', 'home_defensiveWorkRate', 'away_agility', 'home_preferredFoot', 'away_penalties', 'home_power', 'home_penalties', 'home_control', 'away_balance', 'away_preferredFoot', 'home_gkReflexes', 'away_rating', 'away_positioning', 'B365 betting difference', 'home_potential', 'home_crossing', 'BW betting difference', 'home_interceptions', 'home_vision', 'BS betting difference', 'home_jump', 'away_crossing', 'home_strength', 'away_shortPass', 'home_attackingWorkRate', 'SJ betting difference', 'GB betting difference', 'away_acceleration', 'away_gkHandling', 'away_gkReflexes', 'away_jump', 'home_gkDiving', 'away_standingTackle', 'away_longPass', 'away_interceptions', 'away_control', 'away_stamina', 'away_freeKicks', 'away_gkPositioning', 'away_volleys', 'away_slidingTackle', 'PS betting difference', 'away_sprintSpeed', 'away_potential', 'away_dribbling', 'away_defensiveWorkRate', 'away_attackingWorkRate', 'result']
        #fieldnames = ['away_possession', 'home_possession', 'home_shots_on_goal', 'away_shots_on_goal', 'result']
        fieldnames = ['goal_difference', 'result']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        sumTime = 0.
        n = len(matches)
        for index, match in matches.iterrows():
            

            start = time.time()
            phi = extractFeatures(match)
            goalDifference = match['home_team_goal'] - match['away_team_goal']
            if (goalDifference > 0):
                result = 1
            elif (goalDifference == 0):
                result = 0
            else:
                result = -1
            
            if (phi):  
                phi['result'] = result
                writer.writerow(phi)
            sumTime += time.time()-start
            timeElapsed = str(datetime.timedelta(seconds=sumTime))
            averageTime = sumTime/(index+1)
            timeLeft = str(datetime.timedelta(seconds=(n - index)*averageTime))
            print "time elapsed: {} | {} percent done | time left: {}".format(timeElapsed, float(index)/len(matches)*100, timeLeft)
    """
    trainExamples =  matches.sample(100)
    testExamples = matches.sample(20)
    print "Training data to learn weights"
    weights = learnPredictor(trainExamples, testExamples, extractFeatures, numIters=20, eta=0.01)
    print weights
    """


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