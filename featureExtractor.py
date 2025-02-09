#import csv
import numpy as np
import pandas as pd
from collections import defaultdict
from util import *
import csv, time, datetime, re, sys, sqlite3

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
    print "Reading in team data..."
    teams = pd.read_sql_query("select * from Team", conn)
    print "Reading in team attribute data..."
    teamAttributes = pd.read_sql_query("select * from Team_Attributes", conn)
    print "Finished reading in data!"

    return matches, players, playerAttributes, teams, teamAttributes



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

def getTeamAttributes(teamId):
    if (np.isnan(teamId)):
        return {}
    query = "SELECT * FROM 'Team_Attributes' WHERE team_api_id IS {}".format(teamId)
    teams = pd.read_sql_query(query, conn)

    totalAttributes = []
    for index, team in teams.iterrows():
        attributes =  defaultdict(float)
        
        attributes['buildUpPlayPositioningClass'] = 1 if team['buildUpPlayPositioningClass'] == 'Organised' else 0
        attributes['chanceCreationPositioningClass'] = 1 if team['chanceCreationPositioningClass'] == 'Organised' else 0
        attributes['defenceDefenderLineClass'] = 1 if team['defenceDefenderLineClass'] == 'Cover' else 0
        attributes['buildUpPlaySpeed'] = team['buildUpPlaySpeed']
        attributes['buildUpPlayDribbling'] = team['buildUpPlayDribbling']
        attributes['buildUpPlayPassing'] = team['buildUpPlayPassing']
        attributes['chanceCreationPassing'] = team['chanceCreationPassing']
        attributes['chanceCreationCrossing'] = team['chanceCreationCrossing']
        attributes['chanceCreationShooting'] = team['chanceCreationShooting']
        attributes['defencePressure'] = team['defencePressure']
        attributes['defenceAggression'] = team['defenceAggression']
        attributes['defenceTeamWidth'] = team['defenceTeamWidth']
        totalAttributes.append(attributes)

    attributes = averageVectors(totalAttributes)
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


def calculatePrev5Matches(teamId, date, phi, spot):
    assert not np.isnan(teamId)

    query = "SELECT * FROM 'Match' WHERE (home_team_api_id IS {} or away_team_api_id IS {}) AND date < '{}' ORDER BY date DESC Limit 5".format(teamId, teamId, date)
    matches = pd.read_sql_query(query, conn)
    if len(matches) < 5:
        prev5Results = [np.nan for i in range(5)]
    else:
        prev5Results = []
        for index, match in matches.iterrows():
            if match['home_team_goal'] > match['away_team_goal']:
                if match['home_team_api_id'] == teamId:
                    prev5Results.append(1)
                else:
                    prev5Results.append(-1)
            elif match['home_team_goal'] < match['away_team_goal']:
                if match['home_team_api_id'] == teamId:
                    prev5Results.append(-1)
                else:
                    prev5Results.append(1)
            else:
                prev5Results.append(0)
    #[Closest to current date, ..., Farthest]
    i = 0
    for result in prev5Results:
        phi['{}_previous_match_{}_result'.format(spot, i+1)] = result
        i += 1

def head2head(homeTeamId, awayTeamId, phi, date):
    if (np.isnan(homeTeamId) or np.isnan(awayTeamId)):
        return {}
    query = "SELECT * FROM 'Match' WHERE ((home_team_api_id IS {} and away_team_api_id IS {}) or (home_team_api_id IS {} and away_team_api_id IS {})) AND date < '{}' ORDER BY date DESC".format(homeTeamId, awayTeamId, homeTeamId, awayTeamId, date)  
    matches = pd.read_sql_query(query, conn)
    h2h = defaultdict(float)
    h2h['home_head_to_head'] = 0
    h2h['away_head_to_head'] = 0
    if len(matches) == 0:
        h2h['home_head_to_head'] = np.nan
        h2h['away_head_to_head'] = np.nan
    else:
        for index, match in matches.iterrows():
            a, b = match['home_team_api_id'], match['away_team_api_id']
            if a == homeTeamId:
                if match['home_team_goal'] > match['away_team_goal']:
                    h2h['home_head_to_head'] += 1 
                elif match['home_team_goal'] < match['away_team_goal']:
                    h2h['away_head_to_head'] += 1 
            elif b == homeTeamId:
                if match['home_team_goal'] > match['away_team_goal']:
                    h2h['away_head_to_head'] += 1 
                elif match['home_team_goal'] < match['away_team_goal']:
                    h2h['home_head_to_head'] += 1
    combineVectors(phi, h2h, '')

def extractFeatures(match):
    phi = defaultdict(float)

    season = match['season']
    homeTeamID = match['home_team_api_id']
    awayTeamID = match['away_team_api_id']

    calculatePlayerAttributeFeatures(match, phi, season)
    calculateBettingFeatures(match, phi)
    combineVectors(phi, getTeamAttributes(homeTeamID), 'home')
    combineVectors(phi, getTeamAttributes(awayTeamID), 'away')
    calculatePrev5Matches(homeTeamID, match['date'], phi, 'home')
    calculatePrev5Matches(awayTeamID, match['date'], phi, 'away')
    head2head(homeTeamID, awayTeamID, phi, match['date'])


    return phi

def main(matches, players, playerAttributes, teams, teamAttributes):
    with open(fileNameToWriteTo, mode='w') as csv_file:
        fieldnames = extractFeatures(matches.iloc[10000]).keys()
        print "Writing {}-dimensional feature vectors and result to {}".format(len(fieldnames), fileNameToWriteTo)
        fieldnames.append('result')
        
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        sumTime = 0.
        n = len(matches)
        print "Writing {} examples".format(n)
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
            # Print progress
            sys.stdout.write("time elapsed: {} | {} percent done | time left: {}\r".format(timeElapsed, float(index)/len(matches)*100, timeLeft))
            sys.stdout.flush()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: python featureExtractor.py <FILE TO WRITE TO>"
    else:
        fileNameToWriteTo = sys.argv[1]
        print "Writing to {}".format(fileNameToWriteTo)

        matches, players, playerAttributes, teams, teamAttributes = readData()
        main(matches, players, playerAttributes, teams, teamAttributes)
