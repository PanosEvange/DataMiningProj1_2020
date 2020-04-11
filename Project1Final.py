# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_markers: region,endregion
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # <center>Data Mining Project 1 Spring semester 2019-2020</center>
# ## <center>Παναγιώτης Ευαγγελίου &emsp; 1115201500039</center>
# ## <center>Γεώργιος Μαραγκοζάκης &emsp; 1115201500089</center>

# ___

# ### Do all the necessary imports for this notebook

# region
# for data exploration
import pandas as pd
import os
from IPython.display import display

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# endregion

# ## __Data Exploration__

# - ### *Make one csv with the specific columns*

# region
pd.set_option('display.max_columns', 300)
pd.set_option('display.max_rows', 300)

myMonthsFolder = ['febrouary','march','april']
dataPathDir = './data/'

allMonthDf = []

# for each month
for month in myMonthsFolder:
    print('Folder ' + month + ':')
    specificPath = dataPathDir + month + '/'

    # list of dataframes of this month
    monthFilesDfList = []

    # find the column's names of each csv
    for fileName in os.listdir(specificPath):
        # we need to check only .csv files
        if fileName.endswith(".csv"):
            thisCsv = pd.read_csv(os.path.join(specificPath, fileName), dtype='unicode')
            
            # make a list of columns of this file 
            columnsList = [x for x in thisCsv.columns]

            # make columns df for this file
            d = {fileName:columnsList}
            df = pd.DataFrame(data=d)
            
            # append this df to monthFilesDfList
            monthFilesDfList.append(df)
        else:
            continue

    concatDf = pd.concat(monthFilesDfList, axis=1)
    concatDf = concatDf.fillna('-')
    
    display(concatDf)
    allMonthDf.append(concatDf)

# check if corresponding files in months' folders have the same column names
allAreEqual = True
curentDf = allMonthDf[0]
for x in allMonthDf[1:]:
    if not curentDf.equals(x):
        allAreEqual = False
        break

if allAreEqual:
    print('Column names in corresponding files in months\' folders are the same.')
else:
    print('Column names in corresponding files in months\' folders are not the same.')
# endregion

# Παρατηρούμε ότι:
# - Τα αντίστοιχα αρχεία σε όλους τους φακέλους των μηνών έχουν τα ίδια ονόματα στηλών.
# - Τα περισσότερα .csv αρχεία έχουν σχετικά λίγες στήλες εκτός από το αρχείο listings.csv το οποίο
# έχει πολλές στήλες οπότε είναι δύσκολο να βρούμε "με το μάτι" τις στήλες που ζητούνται.

# __Let's find which file has each of the specific columns that we need. As column
# names in corresponding files in months' folders are the same we just need to check only one folder.__

# region
# define the columns that we want to have in train csv
wantedColumns = ['id', 'zipcode', 'transit', 'Bedrooms', 'Beds', 'Review_scores_rating', 'Number_of_reviews', \
                  'Neighbourhood', 'Name', 'Latitude', 'Longitude', 'Last_review', 'Instant_bookable', \
                  'Host_since', 'Host_response_rate', 'Host_identity_verified', 'Host_has_profile_pic', 'First_review', \
                  'Description', 'City', 'cancellation_policy', 'Bed_type', 'Bathrooms', 'Accommodates', 'Amenities', \
                  'Room_type', 'Property_type', 'price', 'Availability_365', 'Minimum_nights']

specificMonthFilesDf = allMonthDf[0]
dfColumns = list(specificMonthFilesDf)

allColumnsFileMapDfList = []

for col in wantedColumns:
    # list of files that include the specific column
    fileLists = []
    
    for fileName in dfColumns:
        # so as not to have problem with case sensitivity
        tempDf = specificMonthFilesDf[fileName].str.upper()

        if col.upper() in tempDf.values:
            fileLists.append(fileName)

    # make df for this col
    d = {col:fileLists}
    df = pd.DataFrame(data=d)

    allColumnsFileMapDfList.append(df)

concatMapDf = pd.concat(allColumnsFileMapDfList, axis=1)
concatMapDf = concatMapDf.fillna('-')

display(concatMapDf)
# endregion
# Παρατηρούμε ότι οι περισσότερες από τις ζητούμενες στήλες βρίσκονται στα αρχεία listings.csv και listings0.csv.
# Οπότε ας δούμε τα περιεχόμενα αυτών των αρχείων.

# region
testDf = pd.read_csv('./data/febrouary/listings0.csv', dtype='unicode')

testDf
# endregion
# region
testDf = pd.read_csv('./data/febrouary/listings.csv', dtype='unicode')

testDf
# endregion


# Από οσο βλέπουμε τα δεδομένα για τις ζητούμενες στήλες είναι ίδια τόσο στο listings0.csv όσο και στο
# στο listings.csv. Οπότε αρκεί να πάρουμε τα δεδομένα από το listings.csv.

# region
# read listings.csv from the 3 folders
febDf = pd.read_csv('./data/febrouary/listings.csv', dtype='unicode')
marDf = pd.read_csv('./data/march/listings.csv', dtype='unicode')
aprDf = pd.read_csv('./data/april/listings.csv', dtype='unicode')

# add extra column for each month
febDf['month'] = ['February' for i in range(febDf.shape[0])]
marDf['month'] = ['March' for i in range(marDf.shape[0])]
aprDf['month'] = ['April' for i in range(aprDf.shape[0])]

monthsDfToConcat = [febDf, marDf, aprDf]
trainCsv = pd.concat(monthsDfToConcat, ignore_index=True)

# let's keep only useful columns
# make all uppercase so as not to have problem with case sensitivity
trainCsv.columns = [x.upper() for x in trainCsv.columns]

# These are the columns we want. We have also added month column 
usefulColumns = ['id', 'zipcode', 'transit', 'Bedrooms', 'Beds', 'Review_scores_rating', 'Number_of_reviews', \
                  'Neighbourhood', 'Name', 'Latitude', 'Longitude', 'Last_review', 'Instant_bookable', \
                  'Host_since', 'Host_response_rate', 'Host_identity_verified', 'Host_has_profile_pic', 'First_review', \
                  'Description', 'City', 'cancellation_policy', 'Bed_type', 'Bathrooms', 'Accommodates', 'Amenities', \
                  'Room_type', 'Property_type', 'price', 'Availability_365', 'Minimum_nights', 'Month']

usefulColumns = [x.upper() for x in usefulColumns]

trainCsv = trainCsv[usefulColumns]

# save to csv file
trainCsv.to_csv('train.csv',index=False)

trainCsv
# endregion

# - ### *Find the most common room type*

# region
# groupBy room_type
roomTypesCountSeries = trainCsv.groupby(['ROOM_TYPE'])['ID'].nunique()

# we need the max of these counts
mostCommonRoomType = roomTypesCountSeries[roomTypesCountSeries == roomTypesCountSeries.max()]

print('The most common room type is the \'' + str(mostCommonRoomType.index[0]) + '\'')
# endregion

# - ### *Price evaluation during 3 months*

# region

# to fill

# endregion

# - ### *Find first 5 neighbourhoods with most reviews*

# region
intDf = trainCsv[['NEIGHBOURHOOD', 'NUMBER_OF_REVIEWS']].copy()
intDf['NUMBER_OF_REVIEWS'] = intDf['NUMBER_OF_REVIEWS'].apply(int)

# groupBy neighbourhood
neighbourhoodReviewsSeries = intDf.groupby(['NEIGHBOURHOOD'])['NUMBER_OF_REVIEWS'].sum()

# sort values
neighbourhoodReviewsSeries = neighbourhoodReviewsSeries.sort_values(ascending=False)

print('The first 5 neighbourhoods with most reviews are:')
for x in neighbourhoodReviewsSeries[0:5].index:
    print(x)
# endregion

# - ### *Find the neighbourhood with the most entries*

# region
# groupBy neighbourhood
neighbourhoodEntriesSeries = trainCsv.groupby(['NEIGHBOURHOOD'])['ID'].nunique()

# sort values
neighbourhoodEntriesSeries = neighbourhoodEntriesSeries.sort_values(ascending=False)

print('The neighbourhood with most reviews is:',neighbourhoodReviewsSeries.index[0])
# endregion

# - ### *Find number of entries per neighbourhood and per month*

# region
# groupBy neighbourhood
neighbourhoodEntriesSeries = trainCsv.groupby(['NEIGHBOURHOOD'])['ID'].nunique()

# groupBy month
monthEntriesSeries = trainCsv.groupby(['MONTH'])['ID'].nunique()
# endregion

sns.set(style="whitegrid")
plt.figure(figsize=(9, 9))
ax = sns.barplot(x=neighbourhoodEntriesSeries.values, y=neighbourhoodEntriesSeries.index)
ax.set(title='Count of entries by neighbourhood', xlabel='Entries', ylabel='Neighbourhood')
plt.show()

sns.set(style="whitegrid")
plt.figure(figsize=(9, 9))
ax = sns.barplot(x=monthEntriesSeries.index, y=monthEntriesSeries.values)
ax.set(title='Count of entries by month', xlabel='Month', ylabel='Entries')
plt.show()

# - ### *Make histogram of neighbourhood variable*

# region

# to fill

# endregion

# - ### *Find the most common room type in every neighbourhood*

# region

# to fill

# endregion

# - ### *Find the most expensive room type*

# region

# to fill

# endregion
