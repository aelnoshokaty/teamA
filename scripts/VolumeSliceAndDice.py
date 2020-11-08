import pandas as pd
import numpy as np
import glob
import os
import sys
import plotly
import matplotlib.pyplot as plt
import numpy
import csv
import pandas

from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

# from dtreeviz.trees import dtreeviz # remember to load the package
# import graphviz

# fix random seed for reproducibility
numpy.random.seed(7)


def loadStates():
    path = os.path.dirname(__file__)
    print(path.rfind('/'))
    path = path[0:path.rfind('/')] + '/rawData/'
    states_region = pd.read_csv(path + 'States_list.csv')
    return states_region


def loadFiles():
    path = os.path.dirname(__file__)
    print(path.rfind('/'))
    path = path[0:path.rfind('/')] + '/rawData/'
    print(path)
    li = []
    all_files = glob.glob(path + "chart_data-volume-days-categories/*.csv")
    states_region = pd.read_csv(path + 'States_list.csv')
    listst = states_region['Abv. '].tolist()
    listrg = states_region['Region'].tolist()
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0, skiprows=10)
        st = filename[-6:-4]
        df['state'] = st
        stateIndex = listst.index(st)
        df['region'] = listrg[stateIndex]
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    df = frame
    newDF = pd.DataFrame()
    newCols = []
    newDF['days'] = df['days']
    newDF['region'] = df['region']
    newDF['state'] = df['state']
    for col in df.columns:
        if col in ['days', 'state', 'region']:
            continue
        else:
            temp = col[0:10]
            newDF[temp] = df[col]

    print(newDF.columns)
    return newDF


def slicAndDice(regions, states, timeAnalysis, fromDate, toDate, DF):
    slicedAndDiced = pd.DataFrame()
    # check on date range
    selectedCols = []
    if fromDate and toDate:
        selected = False
        for col in DF.columns:
            if col in ['days', 'state', 'region'] or selected == True:
                selectedCols.append(col)
            if col == fromDate:
                selectedCols.append(col)
                selected = True
            if col == toDate:
                selected = False
        slicedAndDiced = DF[selectedCols]
    else:
        slicedAndDiced = DF

    slicedAndDiced1 = pd.DataFrame()
    # check on date aggregation
    selectedCols1 = []
    if timeAnalysis == "weekly":
        count = 0
        temp = np.array([0] * 500)
        print(temp)

        ser = pd.Series(temp)
        print(ser)
        for col in slicedAndDiced.columns:
            if col in ['days', 'state', 'region']:
                slicedAndDiced1[col] = slicedAndDiced[col]
            else:
                # start of week
                if count == 0:
                    fromD = col
                    ser = ser + slicedAndDiced[col]
                    count = count + 1
                # accumulate during week
                elif count < 6:
                    ser = ser + slicedAndDiced[col]
                    count = count + 1
                # accumulate during week
                else:
                    # ser.name=fromD+"_"+col
                    ser = ser + slicedAndDiced[col]
                    ser.rename_axis(fromD + "_" + col)
                    slicedAndDiced1[fromD + "_" + col] = ser
                    temp = np.array([0] * 500)
                    ser = pd.Series(temp)
                    count = 0
        # The last week in data
        if count < 6:
            ser.rename_axis(fromD + "_" + col)
            slicedAndDiced1[fromD + "_" + col] = ser

    elif timeAnalysis == "monthly":
        currentMonth = fromDate[5:7]
        beginning = True
        temp = np.array([0] * 500)
        ser = pd.Series(temp)
        print(ser)
        count = 0
        for col in slicedAndDiced.columns:
            # if not the daily volume
            if col in ['days', 'state', 'region']:
                slicedAndDiced1[col] = slicedAndDiced[col]
            # if the daily volume
            else:
                # if the same month accumulate
                if col[5:7] == currentMonth:
                    ser = ser + slicedAndDiced[col]
                    count = count + 1
                # if new month reset
                else:
                    ser.rename_axis(currentMonth)
                    slicedAndDiced1[currentMonth] = ser
                    ser = ser + slicedAndDiced[col]
                    temp = np.array([0] * 500)
                    ser = pd.Series(temp)
                    currentMonth = col[5:7]
                    count = 0
        # last month
        if count > 0:
            ser.rename_axis(currentMonth)
            slicedAndDiced1[currentMonth] = ser

    print(slicedAndDiced1)
    print(slicedAndDiced1['state'])
    print(slicedAndDiced1['region'])
    # slicedAndDiced1.to_csv(path + 'filtered1.csv')

    # check if regions
    dfRegions = slicedAndDiced1
    if regions:
        print("here is choosing from regions")
        # dfRegions=slicedAndDiced1.loc[slicedAndDiced1['region'].isin(regions)]
        dfRegions = slicedAndDiced1[slicedAndDiced1['region'].isin(regions)]

    # check if states
    dfStates = dfRegions
    if states:
        print("here is choosing from states")
        dfStates = dfRegions[dfRegions['state'].isin(states)]

    return dfStates


statesDF = loadStates()
print(statesDF)
filesDF = loadFiles()
print(filesDF)
print(len(filesDF))
print(filesDF['region'])
slicedAndDiced1 = pd.DataFrame()
slicedAndDiced1[['region', 'state']] = filesDF[['region', 'state']]
print(slicedAndDiced1)
# Add the regions and states abbreviations in a list like  ['South','West'] and ['TX','AZ','CA']
# Aggregation is weekly on monthly
# Date format is YYYY-MM-DD

# df=slicAndDice(['South','West'], ['TX','AZ','CA'],'weekly','2020-03-04','2020-09-13',filesDF)
df = slicAndDice(['South', 'West'], ['TX', 'AZ', 'CA'], 'monthly', '2020-03-04', '2020-04-13', filesDF)
path = os.path.dirname(__file__)
print(path.rfind('/'))
path = path[0:path.rfind('/')] + '/rawData/'
df.to_csv(path + 'filtered.csv')

