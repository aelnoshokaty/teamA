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


def loadCOVID():
    path = os.path.dirname(__file__)
    print(path.rfind('/'))
    path = path[0:path.rfind('/')] + '/rawData/'
    COVID = pd.read_csv(path + 'COVIDData.csv')
    return COVID


# This was not used in our research its loading CMU facebook surveys of people wearing masks by time
def loadMaskTime():
    path = os.path.dirname(__file__)
    print(path.rfind('/'))
    path = path[0:path.rfind('/')] + '/rawData/'
    # Averaging CMU facebook surveys of wearing masks by states
    wearing_mask1 = pd.DataFrame()
    wearing_mask = pd.read_csv(
        path + 'fb_survey_wearing_mask_input.csv')  # geo_value , avg --> value, stderr, sample_size
    selectedcols = ['geo_value', 'time_value', 'value', 'stderr', 'sample_size']
    wearing_mask1[selectedcols] = wearing_mask[selectedcols]
    g = {'value': ['mean'], 'stderr': ['mean'], 'sample_size': ['sum']}
    # wearing_mask_grouped = wearing_mask1.groupby(["geo_value","time_value"]).agg(g).reset_index()
    wearing_mask_grouped = wearing_mask1.groupby(["time_value"]).agg(g).reset_index()
    wearing_mask2 = pd.DataFrame()
    # wearing_mask2['geo_value']=wearing_mask_grouped['geo_value']
    wearing_mask2['time_value'] = wearing_mask_grouped['time_value']
    wearing_mask2['sample_size'] = wearing_mask_grouped['sample_size', 'sum']
    wearing_mask2['stderr'] = wearing_mask_grouped['stderr', 'mean']
    wearing_mask2['value'] = wearing_mask_grouped['value', 'mean']
    wearing_mask2.to_csv(path + 'fb_survey_wearing_mask_time.csv')
    print(wearing_mask2)
    return wearing_mask2


# This was not used in our research its loading CMU facebook surveys of people wearing masks
def loadMask():
    path = os.path.dirname(__file__)
    print(path.rfind('/'))
    path = path[0:path.rfind('/')] + '/rawData/'
    # Averaging CMU facebook surveys of wearing masks by states
    wearing_mask1 = pd.DataFrame()
    wearing_mask = pd.read_csv(
        path + 'fb_survey_wearing_mask_input.csv')  # geo_value , avg --> value, stderr, sample_size
    selectedcols = ['geo_value', 'value', 'stderr', 'sample_size']
    wearing_mask1[selectedcols] = wearing_mask[selectedcols]
    g = {'value': ['mean'], 'stderr': ['mean'], 'sample_size': ['mean']}
    wearing_mask_grouped = wearing_mask1.groupby(["geo_value"]).agg(g).reset_index()
    wearing_mask2 = pd.DataFrame()
    wearing_mask2['geo_value'] = wearing_mask_grouped['geo_value']
    wearing_mask2['sample_size'] = wearing_mask_grouped['sample_size', 'mean']
    wearing_mask2['stderr'] = wearing_mask_grouped['stderr', 'mean']
    wearing_mask2['value'] = wearing_mask_grouped['value', 'mean']
    wearing_mask2.to_csv(path + 'fb_survey_wearing_mask.csv')
    print(wearing_mask2)
    return wearing_mask2


# loading the files from Brandwatch for our 10 supervised learning categories on the files of the 50 states aggregated by state
def loadFiles_Agg():
    path = os.path.dirname(__file__)
    print(path.rfind('/'))
    path = path[0:path.rfind('/')] + '/rawData/'
    print(path)
    li = []
    all_files = glob.glob(path + "chart_data-volume-days-categories/*.csv")

    # finish grouping wearing masks

    states_region = pd.read_csv(path + 'States_list.csv')
    COVID = pd.read_csv(path + 'COVIDData.csv')
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
    g = {}
    for col in newDF.columns:
        if col in ['region', 'days', 'state']:
            if col == "state":
                continue
            g[col] = ['max']
        else:
            g[col] = ['sum']

    newDF_grouped = newDF.groupby(["state"]).agg(g).reset_index()
    print(newDF_grouped.columns)
    print(newDF_grouped)
    newDF1 = pd.DataFrame()
    for col in newDF_grouped.columns:
        if col[0] == 'days':
            continue
        newDF1[col[0]] = newDF_grouped[col]
    print(newDF1.columns)
    newDF1.to_csv('newDF_grouped.csv')
    return newDF1


# loading the files from Brandwatch for our 10 supervised learning categories on the files of the 50 states
def loadFiles():
    path = os.path.dirname(__file__)
    print(path.rfind('/'))
    path = path[0:path.rfind('/')] + '/rawData/'
    print(path)
    li = []
    all_files = glob.glob(path + "chart_data-volume-days-categories/*.csv")

    # finish grouping wearing masks

    states_region = pd.read_csv(path + 'States_list.csv')
    COVID = pd.read_csv(path + 'COVIDData.csv')
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
    newDF.to_csv("TweetsDaily.csv")
    return newDF


# slice and dice negative tweets by from or to date, select certain region or certain states. Aggregated monthly or weekly
def slicAndDice(regions, states, timeAnalysis, fromDate, toDate, DF):
    path = os.path.dirname(__file__)
    print(path.rfind('/'))
    path = path[0:path.rfind('/')] + '/rawData/'
    slicedAndDiced = pd.DataFrame()
    # check on date range
    selectedCols = []
    if fromDate and toDate:
        selected = False
        for col in DF.columns:
            if col == 'days':
                selectedCols.append('days')
            if col == 'state':
                selectedCols.append(col)
            if col == 'region':
                selectedCols.append(col)
            if selected == True:
                selectedCols.append(col)
            if col == fromDate:
                selectedCols.append(col)
                selected = True
            if col == toDate:
                selected = False

        slicedAndDiced = DF[selectedCols]
        print(selectedCols)


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
                    ser.rename_axis(fromD[5:10] + " - " + col[5:10])
                    slicedAndDiced1[fromD[5:10] + " - " + col[5:10]] = ser
                    temp = np.array([0] * 500)
                    ser = pd.Series(temp)
                    count = 0
        # The last week in data
        if count < 6:
            ser.rename_axis(fromD[5:10] + " - " + col[5:10])

    elif timeAnalysis == "monthly":
        currentMonth = fromDate[5:7]
        beginning = True
        temp = np.array([0] * 500)
        ser = pd.Series(temp)

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

    dfStates.to_csv(path + 'filtered.csv')


# slice and dice negative tweets and COVID cases by from or to date, select certain region or certain states. Aggregated monthly or weekly
def slicAndDice_COVID(regions, states, timeAnalysis, fromDate, toDate, DF, COVID_DF):
    path = os.path.dirname(__file__)
    print(path.rfind('/'))
    path = path[0:path.rfind('/')] + '/rawData/'
    slicedAndDiced = pd.DataFrame()
    COVID_slicedAndDiced = pd.DataFrame()
    # check on date range
    selectedCols = []
    COVID_selectedCols = []
    if fromDate and toDate:
        selected = False
        for col in DF.columns:
            if col == 'days':
                selectedCols.append('days')
            if col == 'state':
                selectedCols.append(col)
                COVID_selectedCols.append(col)
            if col == 'region':
                selectedCols.append(col)
                COVID_selectedCols.append(col)
            if selected == True:
                selectedCols.append(col)
                COVID_selectedCols.append(col[0:10])
            if col == fromDate:
                selectedCols.append(col)
                COVID_selectedCols.append(col[0:10])
                selected = True
            if col == toDate:
                selected = False
                COVID_selectedCols.append(col[0:10])
        slicedAndDiced = DF[selectedCols]
        print(COVID_DF.columns)
        print(selectedCols)
        print(COVID_selectedCols)
        for col in COVID_selectedCols:
            if col in ['region', 'state']:
                COVID_slicedAndDiced[col] = COVID_DF[col]
            else:
                COVID_slicedAndDiced[col] = COVID_DF[str(int(col[5:7])) + '/' + str(int(col[8:10])) + '/' + '20']

        print(COVID_slicedAndDiced)


    else:
        slicedAndDiced = DF
        COVID_slicedAndDiced = COVID_DF

    slicedAndDiced1 = pd.DataFrame()
    COVID_slicedAndDiced1 = pd.DataFrame()
    # check on date aggregation
    selectedCols1 = []
    if timeAnalysis == "weekly":
        count = 0
        temp = np.array([0] * 50)
        COIVD_temp = np.array([0] * 50)
        print(temp)
        ser = pd.Series(temp)
        COVID_ser = pd.Series(COIVD_temp)
        print(ser)

        for col in slicedAndDiced.columns:
            if col in ['days', 'state', 'region']:
                slicedAndDiced1[col] = slicedAndDiced[col]
                if col in ['state', 'region']:
                    COVID_slicedAndDiced1[col] = slicedAndDiced[col]

            else:
                # start of week
                if count == 0:
                    fromD = col
                    ser = ser + slicedAndDiced[col]
                    COVID_ser = COVID_ser + COVID_slicedAndDiced[col[0:10]]
                    count = count + 1
                # accumulate during week
                elif count < 6:
                    ser = ser + slicedAndDiced[col]
                    COVID_ser = COVID_ser + COVID_slicedAndDiced[col[0:10]]
                    count = count + 1
                # accumulate during week
                else:
                    # ser.name=fromD+"_"+col
                    ser = ser + slicedAndDiced[col]
                    ser.rename_axis(fromD[5:10] + " - " + col[5:10])
                    slicedAndDiced1[fromD[5:10] + " - " + col[5:10]] = ser
                    temp = np.array([0] * 500)
                    ser = pd.Series(temp)

                    COVID_ser = COVID_ser + COVID_slicedAndDiced[col]
                    COVID_ser.rename_axis(fromD[5:10] + " - " + col[5:10])
                    COVID_slicedAndDiced1[fromD[5:10] + " - " + col[5:10]] = COVID_ser
                    COIVD_temp = np.array([0] * 50)
                    COVID_ser = pd.Series(COIVD_temp)
                    count = 0
        # The last week in data
        if count < 6:
            ser.rename_axis(fromD[5:10] + " - " + col[5:10])
            slicedAndDiced1[fromD[5:10] + " - " + col[5:10]] = ser

            COVID_ser.rename_axis(fromD[5:10] + " - " + col[5:10])
            COVID_slicedAndDiced1[fromD[5:10] + " - " + col[5:10]] = COVID_ser

    elif timeAnalysis == "monthly":
        currentMonth = fromDate[5:7]
        beginning = True
        temp = np.array([0] * 50)
        ser = pd.Series(temp)

        COIVD_temp = np.array([0] * 50)
        COVID_ser = pd.Series(COIVD_temp)
        print(COVID_ser)
        count = 0
        for col in slicedAndDiced.columns:
            # if not the daily volume
            if col in ['days', 'state', 'region']:
                if col in ['state', 'region']:
                    COVID_slicedAndDiced1[col] = COVID_slicedAndDiced[col]
                slicedAndDiced1[col] = slicedAndDiced[col]
            # if the daily volume
            else:
                # if the same month accumulate
                if col[5:7] == currentMonth:
                    ser = ser + slicedAndDiced[col]
                    COVID_ser = COVID_ser + COVID_slicedAndDiced[col]
                    count = count + 1
                # if new month reset
                else:
                    ser.rename_axis(currentMonth)
                    slicedAndDiced1[currentMonth] = ser
                    ser = ser + slicedAndDiced[col]
                    temp = np.array([0] * 50)
                    ser = pd.Series(temp)

                    COVID_ser.rename_axis(currentMonth)
                    COVID_slicedAndDiced1[currentMonth] = COVID_ser
                    COVID_ser = COVID_ser + COVID_slicedAndDiced[col]
                    COVID_temp = np.array([0] * 50)
                    COVID_ser = pd.Series(COVID_temp)

                    currentMonth = col[5:7]
                    count = 0
        # last month
        if count > 0:
            ser.rename_axis(currentMonth)
            slicedAndDiced1[currentMonth] = ser

            COVID_ser.rename_axis(currentMonth)
            COVID_slicedAndDiced1[currentMonth] = COVID_ser

    print(slicedAndDiced1)
    print(slicedAndDiced1['state'])
    print(slicedAndDiced1['region'])
    # slicedAndDiced1.to_csv(path + 'filtered1.csv')

    # check if regions
    COVID_dfRegions = COVID_slicedAndDiced1
    dfRegions = slicedAndDiced1
    if regions:
        print("here is choosing from regions")
        # dfRegions=slicedAndDiced1.loc[slicedAndDiced1['region'].isin(regions)]
        dfRegions = slicedAndDiced1[slicedAndDiced1['region'].isin(regions)]
        COVID_dfRegions = COVID_slicedAndDiced1[COVID_slicedAndDiced1['region'].isin(regions)]

    # check if states
    dfStates = dfRegions
    COVID_dfStates = COVID_dfRegions
    if states:
        print("here is choosing from states")
        dfStates = dfRegions[dfRegions['state'].isin(states)]
        COVID_dfStates = COVID_dfRegions[COVID_dfRegions['state'].isin(states)]

    COVID_dfStates.to_csv(path + 'COVID_filtered.csv')
    dfStates.to_csv(path + 'filtered.csv')


# Getting the time series correlation between COVID and negative tweets

import pandas as pd
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

path = os.path.dirname(__file__)
path = path[0:path.rfind('/')] + '/rawData/'
# df = pd.read_csv(path+'synchrony_sample.csv')
df = pd.read_csv(path + 'synchrony_sample_masks.csv')
overall_pearson_r = df.corr().iloc[0, 1]
print(f"Pandas computed Pearson r: {overall_pearson_r}")
# out: Pandas computed Pearson r: 0.7732132385329556

# r, p = stats.pearsonr(df.dropna()['COVID'], df.dropna()['Tweets'])
r, p = stats.pearsonr(df.dropna()['NotWearing'], df.dropna()['Tweets'])

print(f"Scipy computed Pearson r: {r} and p-value: {p}")
# out: Scipy computed Pearson r: 0.7732132385329555 and p-value: 6.292525985227903e-57

# Compute rolling window synchrony
f, ax = plt.subplots(figsize=(7, 3))
df.rolling(window=30, center=True).median().plot(ax=ax)
ax.set(xlabel='Time', ylabel='Pearson r')
ax.set(title=f"Overall Pearson r = {np.round(overall_pearson_r, 2)}");
plt.savefig(path + 'Pearson.png', bbox_inches='tight')


# Getting the time lag/lead between the waves of COVID and negative tweets

def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))


d1 = df['Tweets']
# d2 = df['COVID']
d2 = df['NotWearing']

seconds = 5
fps = 30
rs = [crosscorr(d1, d2, lag) for lag in range(-int(seconds * fps), int(seconds * fps + 1))]
offset = np.ceil(len(rs) / 2) - np.argmax(rs)
f, ax = plt.subplots(figsize=(14, 3))
ax.plot(rs)
ax.axvline(np.ceil(len(rs) / 2), color='k', linestyle='--', label='Center')
ax.axvline(np.argmax(rs), color='r', linestyle='--', label='Peak synchrony')
# ax.set(title=f'Offset = {offset} frames\nTweets leads <> COVID leads', ylim=[.1, .81], xlim=[0, 301], xlabel='Offset',
# ylabel='Pearson r')
ax.set(title=f'Offset = {offset} frames\nTweets leads <> Not Wearing Masks leads', ylim=[.1, .81], xlim=[0, 301],
       xlabel='Offset',
       ylabel='Pearson r')
ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150]);
plt.legend()
plt.savefig(path + 'lead.png', bbox_inches='tight')

loadMaskTime()
statesDF = loadStates()
print(statesDF)
filesDF = loadFiles()
COVID_DF = loadCOVID()
filesDF = loadFiles_Agg()
print(filesDF)
print(len(filesDF))
print(filesDF['region'])
slicedAndDiced1 = pd.DataFrame()
slicedAndDiced1[['region', 'state']] = filesDF[['region', 'state']]
print(slicedAndDiced1)
# Add the regions and states abbreviations in a list like  ['South','West'] and ['TX','AZ','CA']
# Aggregation is weekly on monthly
# Date format is YYYY-MM-DD

# df=slicAndDice_COVID(['South','West','Northeast','Midwest'], ['AL',	'AK',	'AZ',	'AR',	'CA',	'CO',	'CT',	'DE',	'FL',	'GA',	'HI',	'ID',	'IL',	'IN',	'IA',	'KS',	'KY',	'LA',	'ME',	'MD',	'MA',	'MI',	'MN',	'MS',	'MO',	'MT',	'NE',	'NV',	'NH',	'NJ',	'NM',	'NY',	'NC',	'ND',	'OH',	'OK',	'OR',	'PA',	'RI',	'SC',	'SD',	'TN',	'TX',	'UT',	'VT',	'VA',	'WA',	'WV',	'WI',	'WY',],'weekly','2020-01-22','2020-10-27',filesDF,COVID_DF)
df = slicAndDice_COVID(['South', 'West', 'Northeast', 'Midwest'],
                       ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS',
                        'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY',
                        'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV',
                        'WI', 'WY', ], 'monthly', '2020-01-22', '2020-10-27', filesDF, COVID_DF)
# df=slicAndDice(['South','West','Northeast','Midwest'], ['AL',	'AK',	'AZ',	'AR',	'CA',	'CO',	'CT',	'DE',	'FL',	'GA',	'HI',	'ID',	'IL',	'IN',	'IA',	'KS',	'KY',	'LA',	'ME',	'MD',	'MA',	'MI',	'MN',	'MS',	'MO',	'MT',	'NE',	'NV',	'NH',	'NJ',	'NM',	'NY',	'NC',	'ND',	'OH',	'OK',	'OR',	'PA',	'RI',	'SC',	'SD',	'TN',	'TX',	'UT',	'VT',	'VA',	'WA',	'WV',	'WI',	'WY',],'weekly','2020-01-22','2020-10-27',filesDF,COVID_DF)
# df=slicAndDice(['South','West'], ['TX','AZ','CA'],'weekly','2020-03-04','2020-09-13',filesDF)
# df=slicAndDice(['South','West'], ['TX','AZ','CA'],'weekly','2020-03-04','2020-03-25',filesDF)
# df=slicAndDice_COVID(['South','West'], ['TX','AZ','CA'],'weekly','2020-03-04','2020-03-25',filesDF,COVID_DF)
# df=slicAndDice(['South','West'], ['TX','AZ','CA'],'monthly','2020-03-04','2020-04-13',filesDF)


# 3.2.1. Facebook Surveys
# We want to compare the volume of tweets against wearing masks and the actual people not wearing masks, we used Carnegie Mellon university CMU Facebook surveys (https://delphi.cmu.edu/covidcast) to collect data about people wearing masks. CMU surveyed on Facebook from 8 September to
