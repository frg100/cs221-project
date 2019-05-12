import csv
import pandas as pd
import sys
import numpy as np

fileName = 'basicMatchData.csv'
writeTo = 'cleanedBasicMatchData'


def readData(fileName):
    print "Reading in match data..."
    df = pd.read_csv(fileName)
    print "Finished reading in data!"
    return df


def calculateAverages(df):
    labels = list(df)
    accumulator = { label: { 'sum': 0, 'count': 0 } for label in labels }


    for index, row in df.iterrows():
        for label in labels:
            value = row[label]
            if pd.notna(value):
                accumulator[label]['sum'] += value
                accumulator[label]['count'] += 1
        if index % 10 == 0:  
            sys.stdout.write("{} percent done [{}/{}]\r".format(((float(index)/len(df))*100), index, len(df)))
            sys.stdout.flush()

    print "Finished processing the data!"
    averages = {label: accumulator[label]['sum']/ accumulator[label]['count'] for label in labels}
    return averages


def insertAverages(df, averages):
    return df.replace(to_replace = np.nan, value=averages, inplace=False)


if __name__ == '__main__':
    df = readData(fileName)
    averages = calculateAverages(df)
    cleanDF = insertAverages(df, averages)
    cleanDF.to_csv(writeTo, encoding='utf-8', index=False)



