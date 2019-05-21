import csv
import pandas as pd
import sys
import numpy as np
import math


def readData(fileName):
    print "Reading in match data..."
    df = pd.read_csv(fileName)
    print "Finished reading in data!"
    return df


def calculateAverages(df):
    labels = list(df)[:-1]
    accumulator = { label: { 'sum': 0, 'count': 1 } for label in labels }


    for index, row in df.iterrows():
        for label in labels:
            value = row[label]
            if pd.notna(value):
                accumulator[label]['sum'] += value
                accumulator[label]['count'] += 1
        if index % 10 == 0:  
            sys.stdout.write("[averages] | {} percent done [{}/{}]\r".format(((float(index)/len(df))*100), index, len(df)))
            sys.stdout.flush()

    print "Finished processing the data!"
    averages = {label: accumulator[label]['sum']/ accumulator[label]['count'] for label in labels}
    return averages


def calculateStandardDeviations(df, averages):
    labels = list(df)[:-1]
    accumulator = { label: { 'sum': 0, 'count': 0 } for label in labels }


    for index, row in df.iterrows():
        for label in labels:
            value = row[label]
            if pd.notna(value):
                accumulator[label]['sum'] += (value - averages[label])**2
                accumulator[label]['count'] += 1
        if index % 10 == 0:  
            sys.stdout.write("[std dev] | {} percent done [{}/{}]\r".format(((float(index)/len(df))*100), index, len(df)))
            sys.stdout.flush()

    print "Finished processing the data!"
    averages = {label: math.sqrt(accumulator[label]['sum']/ accumulator[label]['count']) for label in labels}
    return averages


def insertAverages(df, averages):
    return df.replace(to_replace = np.nan, value=averages, inplace=False)


def standardize(df, averages, standardDeviations):
    """
    This function takes each value, subtracts the mean, and divides by the standard deviation
    """
    labels = list(df)[:-1]

    for index, row in df.iterrows():
        for label in labels:
            value = row[label]
            newValue = (value - averages[label])/standardDeviations[label]
            df.at[index, label] = newValue
        if index % 10 == 0:  
            sys.stdout.write("[standardize] | {} percent done [{}/{}]\r".format(((float(index)/len(df))*100), index, len(df)))
            sys.stdout.flush()
    print "Finished processing the data!"
    return df




if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: python standardizeData.py <source> <destination>"
    else:
        fileName = sys.argv[1]
        writeTo = sys.argv[2]
        print "Reading data from {} and writing it to {}".format(fileName, writeTo)

        df = readData(fileName)
        averages = calculateAverages(df)
        cleanDF = insertAverages(df, averages)
        standardDeviations = calculateStandardDeviations(cleanDF, averages)
        standardizedDF = standardize(cleanDF, averages, standardDeviations)
        standardizedDF.to_csv(writeTo, encoding='utf-8', index=False)



