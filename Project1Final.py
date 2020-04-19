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
from IPython.display import Image
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# for preprocessing
from string import punctuation

# for recommendation system
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

# replace last_review date with its review_text
reviewsCsvs = ['./data/febrouary/reviews.csv', './data/march/reviews.csv', './data/april/reviews.csv']
monthsDfs = [febDf, marDf, aprDf]

for i in range(0,len(monthsDfs)):
    currentReviewDf = pd.read_csv(reviewsCsvs[i], dtype='unicode')

    # we need only the 2 key-columns (listing_id, date) and the "new" column (comments)
    currentReviewDf = currentReviewDf[['listing_id','date','comments']]

    #rename key-columns so as to be same in both dataframes
    currentReviewDf = currentReviewDf.rename(columns={"date": "review_date", "comments": "review_text"})
    monthsDfs[i] = monthsDfs[i].rename(columns={"id": "listing_id", "last_review": "review_date"})

    #merge dataframes on listing_id and review_date (which is the last_review date in main "train" csv)
    monthsDfs[i] = pd.merge(monthsDfs[i], currentReviewDf, how='left', on=['listing_id', 'review_date'])

    #revert to previous names in main "train" csv
    monthsDfs[i] = monthsDfs[i].rename(columns={"listing_id": "id", "review_text": "last_review"})

    #we may have some duplicates, because there may be more than one review for the last_review date
    #so we will keep only the first one
    monthsDfs[i] = monthsDfs[i][monthsDfs[i].duplicated(subset='id', keep='first') == False]

trainCsv = pd.concat(monthsDfs, ignore_index=True)

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

# - ### *Handling Missing Data*

# Let's see if we have missing data

# Let's check which columns contain nan values
trainCsv.isna().any()

#   - #### Handling missing data on zipcode column

# region
# find unique values of neighboorhood variable
neighborhoodNames = list(trainCsv.NEIGHBOURHOOD.unique())

# remove nan values
neighborhoodNames = [x for x in neighborhoodNames if not pd.isna(x)]

# group by neighborhood
groupedByNeighborhood = trainCsv.groupby(['NEIGHBOURHOOD'])

# for each neighborhood find the most common zipcode
zipCodeNeighborhoodDict = dict()

for neighborhood in neighborhoodNames:
    zipCodeCountSeries = groupedByNeighborhood.get_group(neighborhood).groupby('ZIPCODE')['ID'].nunique()
    mostCommonZipCode = zipCodeCountSeries[zipCodeCountSeries == zipCodeCountSeries.max()]
    zipCodeNeighborhoodDict[neighborhood] = mostCommonZipCode.index[0]

# fill nan values with the most common zipcode of the corresponding neighborhood
trainCsv['ZIPCODE'] = trainCsv['ZIPCODE'].fillna(trainCsv.NEIGHBOURHOOD.map(zipCodeNeighborhoodDict))
# endregion

#   - #### Handling missing data on transit column

# fill nan values with empty text, as transit column has free text content
trainCsv['TRANSIT'] = trainCsv['TRANSIT'].fillna("")

#   - #### Handling missing data on bedroom column


# Let's check how many entries have nan value on bedroom column

len(trainCsv[trainCsv['BEDROOMS'].isna() == True])

# region
# the bedrooms nan values are few, so we will fill nan values with the number of beds, that means 1 bedroom for each bed
trainCsv['BEDROOMS'] = trainCsv['BEDROOMS'].fillna(trainCsv['BEDS'])

# drop rows that have nan values on both bedrooms and beds columns (where bedrooms are still nan)
trainCsv.dropna(subset=['BEDROOMS'], inplace=True)
# endregion

#   - #### Handling missing data on beds column

# Let's check how many entries have nan value on beds column

len(trainCsv[trainCsv['BEDS'].isna() == True])

# the beds nan values are few, so we will fill nan values with the number of bedrooms, that means 1 bedroom for each bed
trainCsv['BEDS'] = trainCsv['BEDS'].fillna(trainCsv['BEDROOMS'])

#   - #### Handling missing data on review scores rating column

# most of entries with nan value on REVIEW_SCORES_RATING has 0 number of reviews, so we will fill REVIEW_SCORES_RATING with '-'
# because 0 score rating means that they are bad hosts, but we don't know if they are good or bad hosts as they don't have many reviews
trainCsv['REVIEW_SCORES_RATING'] = trainCsv['REVIEW_SCORES_RATING'].fillna('-')

#   - #### Handling missing data on neighborhood column

# Let's check how many entries have nan value on neighbourhood column

len(trainCsv[trainCsv['NEIGHBOURHOOD'].isna() == True])

# the number of these entries is not large so we will drop them
trainCsv.dropna(subset=['NEIGHBOURHOOD'], inplace=True)

#   - #### Handling missing data on name column

# fill nan values with empty text, as name column has free text content
trainCsv['NAME'] = trainCsv['NAME'].fillna("")

#   - #### Handling missing data on last review column

# Let's check how many entries with number of reviews != 0, have nan values on last review column

len(trainCsv[(trainCsv['LAST_REVIEW'].isna() == True) & (trainCsv['NUMBER_OF_REVIEWS'] != '0')])

# As we see the number of these entries is not large, so we will drop these rows and we will fill
# nan values of the rows with number of reviews == 0 with '' 

# region
# fill nan values of entries that don't have any reviews with ''
trainCsv['LAST_REVIEW'] = trainCsv.apply(
    lambda row: '' if (pd.isna(row['LAST_REVIEW']) and row['NUMBER_OF_REVIEWS'] == '0') else row['LAST_REVIEW'],
    axis=1
)

# drop entries with nan on last_review that have number_of_reviews > 0 
trainCsv.dropna(subset=['LAST_REVIEW'], inplace=True)
# endregion

#   - #### Handling missing data on host since column

# Let's check how many entries have nan value on host since column

len(trainCsv[trainCsv['HOST_SINCE'].isna() == True])

# the number of these entries is not large so we will drop them
trainCsv.dropna(subset=['HOST_SINCE'], inplace=True)

#   - #### Handling missing data on host response rate column

# Let's check how many entries have nan value on host repsonse rate column

len(trainCsv[trainCsv['HOST_RESPONSE_RATE'].isna() == True])

# The number of these entries is large so we cannot drop them

# Let's fill nan values with '-' as we cannot retrieve the response rate from something else

trainCsv['HOST_RESPONSE_RATE'] = trainCsv['HOST_RESPONSE_RATE'].fillna('-')

#   - #### Handling missing data on host identity verified column

# Let's check how many entries have nan value on host identity verified column

len(trainCsv[trainCsv['HOST_IDENTITY_VERIFIED'].isna() == True])

# Now there are no nan values for this column (as we have dropped some rows in previous steps) 

#   - #### Handling missing data on host has profile pic column

# Let's check how many entries have nan value on host has profile pic column

len(trainCsv[trainCsv['HOST_HAS_PROFILE_PIC'].isna() == True])

# Now there are no nan values for this column (as we have dropped some rows in previous steps)

#   - #### Handling missing data on first review column

# Let's check how many entries have nan value on first review  column

len(trainCsv[trainCsv['FIRST_REVIEW'].isna() == True])

# Let's check how many entries with number of reviews != 0, have nan values on first review column

len(trainCsv[(trainCsv['FIRST_REVIEW'].isna() == True) & (trainCsv['NUMBER_OF_REVIEWS'] != '0')])

# There is no nan value on first review column with number of reviews != 0

# Let's fill nan values with '-' as there is no first review (because these entries have 0 reviews)

trainCsv['FIRST_REVIEW'] = trainCsv['FIRST_REVIEW'].fillna('-')

#   - #### Handling missing data on description column

# Let's check how many entries have nan value on description  column

len(trainCsv[trainCsv['DESCRIPTION'].isna() == True])

# Fill nan values with empty text, as description column has free text content

trainCsv['DESCRIPTION'] = trainCsv['DESCRIPTION'].fillna("")

#   - #### Handling missing data on city column

# Let's check how many entries have nan value on city  column

len(trainCsv[trainCsv['CITY'].isna() == True])

# The number of these entries is not large so we will drop them

trainCsv.dropna(subset=['CITY'], inplace=True)

# reset index of dataframe as we have dropped some rows
trainCsv.reset_index(inplace=True, drop=True)

# Let's see if we still have missing data

trainCsv.isna().any()

# - ### *Find the most common room type*

# region
# groupBy room_type
roomTypesCountSeries = trainCsv.groupby(['ROOM_TYPE'])['ID'].nunique()

# we need the max of these counts
mostCommonRoomType = roomTypesCountSeries[roomTypesCountSeries == roomTypesCountSeries.max()]

print('The most common room type is the \'' + str(mostCommonRoomType.index[0]) + '\'')

sns.set(style="whitegrid")
plt.figure(figsize=(9, 9))
ax = sns.barplot(x=roomTypesCountSeries.values, y=roomTypesCountSeries.index)
ax.set(title='Histogram of Room Type variable', xlabel='Entries', ylabel='Room Type')
plt.show()
# endregion

# - ### *Price evaluation during 3 months*

# region
floatDf = trainCsv[['PRICE', 'MONTH']].copy()
floatDf['PRICE'] = floatDf['PRICE'].str.replace('$','')
floatDf['PRICE'] = floatDf['PRICE'].str.replace(',','')
floatDf['PRICE'] = floatDf['PRICE'].astype(float)

# replace Months with number so as we can sort them 
d = {'February':2, 'March':3, 'April':4}

floatDf.MONTH = floatDf.MONTH.map(d)

# groupBy month
pricesSeries = floatDf.groupby(['MONTH'])['PRICE'].mean()

pricesSeries = pricesSeries.sort_values(ascending=False)

# replace again the number with monthname 
dInverse = {2:'February', 3:'March', 4:'April'}

pricesSeries = pricesSeries.rename(dInverse)

sns.set(style="whitegrid")
plt.figure(figsize=(9, 9))
plt.plot(pricesSeries.index, pricesSeries.values)
plt.xlabel('Month')
plt.ylabel('Price')
plt.title('Price evaluation during 3 months')
plt.show()
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

neighbourhoodReviewsSeries = neighbourhoodReviewsSeries.head(5)
sns.set(style="whitegrid")
plt.figure(figsize=(9, 9))
ax = sns.barplot(x=neighbourhoodReviewsSeries.values, y=neighbourhoodReviewsSeries.index)
ax.set(title='Histogram of Neighbourhood Reviews variable', xlabel='Entries', ylabel='Neighbourhood')
plt.show()
# endregion

# - ### *Find the neighbourhood with the most entries*

# region
# groupBy neighbourhood
neighbourhoodEntriesSeries = trainCsv.groupby(['NEIGHBOURHOOD'])['ID'].nunique()

# sort values
neighbourhoodEntriesSeries = neighbourhoodEntriesSeries.sort_values(ascending=False)

print('The neighbourhood with most entries is:',neighbourhoodReviewsSeries.index[0])

sns.set(style="whitegrid")
plt.figure(figsize=(9, 9))
ax = sns.barplot(x=neighbourhoodEntriesSeries.values, y=neighbourhoodEntriesSeries.index)
ax.set(title='Histogram of Neighbourhood variable', xlabel='Entries', ylabel='Neighbourhood')
plt.show()
# endregion

# - ### *Find number of entries per neighbourhood and per month*

sns.set(style="whitegrid")
plt.figure(figsize=(10, 13))
ax = sns.countplot(y="NEIGHBOURHOOD", hue="MONTH", data=trainCsv)
ax.set(title='Count of entries by month and by neighborhood', xlabel='Entries', ylabel='Neighborhood')
plt.show()

# - ### *Make histogram of neighbourhood variable*

# region
# groupBy neighbourhood
neighbourhoodEntriesSeries = trainCsv.groupby(['NEIGHBOURHOOD'])['ID'].nunique()

sns.set(style="whitegrid")
plt.figure(figsize=(9, 9))
ax = sns.barplot(x=neighbourhoodEntriesSeries.values, y=neighbourhoodEntriesSeries.index)
ax.set(title='Histogram of Neighbourhood variable', xlabel='Entries', ylabel='Neighbourhood')
plt.show()
# endregion

# - ### *Find the most common room type in every neighbourhood*

# region

copyTrainCsv = trainCsv[['NEIGHBOURHOOD', 'ROOM_TYPE', 'ID']].copy()
copyTrainCsv = copyTrainCsv[copyTrainCsv.duplicated(subset='ID', keep='first') == False]

sns.set(style="whitegrid")
plt.figure(figsize=(10, 13))
ax = sns.countplot(y="NEIGHBOURHOOD", hue="ROOM_TYPE", data=copyTrainCsv)
ax.set(title='Count of entries by room type in every neighbourhood', xlabel='Entries', ylabel='Neighborhood')
plt.show()
# endregion

# - ### *Find the most expensive room type*

# region

floatDf = trainCsv[['ROOM_TYPE', 'PRICE']].copy()
floatDf['PRICE'] = floatDf['PRICE'].str.replace('$','')
floatDf['PRICE'] = floatDf['PRICE'].str.replace(',','')
floatDf['PRICE'] = floatDf['PRICE'].astype(float)

# groupBy month
pricesSeries = floatDf.groupby(['ROOM_TYPE'])['PRICE'].mean()
mostExpensiveRoomType = pricesSeries[pricesSeries == pricesSeries.max()]
print('The most expensive room type is the \'' + str(mostExpensiveRoomType.index[0]) + '\'')

sns.set(style="whitegrid")
plt.figure(figsize=(9, 9))
ax = sns.barplot(x=pricesSeries.values, y=pricesSeries.index)
ax.set(title='Histogram of room type mean price variable', xlabel='Entries', ylabel='Room type')
plt.show()
# endregion

# - ### *Show some entries in Follium Map*

# region
# Let's select month February
subsetTrainCsv = trainCsv[trainCsv.MONTH == 'February']

# Gather locations
locations = subsetTrainCsv[['LATITUDE', 'LONGITUDE']]
locationlist = locations.values.tolist()

map = folium.Map(location=[37.97615090257737, 23.72510962013185], zoom_start=14)
tooltip = 'Click me!'

# for point in range(0, len(locationlist)):
for point in range(0, 500):
    stringForPopUp = subsetTrainCsv['PROPERTY_TYPE'][point] + '\n' + subsetTrainCsv['ROOM_TYPE'][point] \
        + '\n' + subsetTrainCsv['BED_TYPE'][point] + '\n' + subsetTrainCsv['PRICE'][point] \
        + '\n'
    folium.Marker(locationlist[point], popup=stringForPopUp, tooltip=tooltip).add_to(map)

map
# endregion

# - ### *Wordclouds*

#   - #### Neigbourhood Wordcloud

# region
wholeNeihborhoodText = ''
for neihborhoodText in trainCsv['NEIGHBOURHOOD']:  
    # make words like "Agios Nikolaos" one word -> AgiosNikolaos
    neihborhoodText = neihborhoodText.replace(" ", "")
    
    wholeNeihborhoodText = wholeNeihborhoodText + ' ' + neihborhoodText

wc = WordCloud(width=600, height=600, background_color='white',collocations = False, stopwords=ENGLISH_STOP_WORDS)

wc.generate(wholeNeihborhoodText)
wc.to_file('neighbourhoodWordcloud.png')

Image('neighbourhoodWordcloud.png')
# endregion
#   - #### Transit Wordcloud

# region
wholeTransitText = ''
for transitText in trainCsv['TRANSIT']:  
    wholeTransitText = wholeTransitText + ' ' + transitText

wc = WordCloud(width=600, height=600, background_color='white', stopwords=ENGLISH_STOP_WORDS)

wc.generate(wholeTransitText)
wc.to_file('transitWordcloud.png')

Image('transitWordcloud.png')
# endregion

#   - #### Description Wordcloud

# region
wholeDescriptionText = ''
for descriptionText in trainCsv['DESCRIPTION']:    
    wholeDescriptionText = wholeDescriptionText + ' ' + descriptionText

wc = WordCloud(width=600, height=600, background_color='white', stopwords=ENGLISH_STOP_WORDS)

wc.generate(wholeDescriptionText)
wc.to_file('descriptionWordcloud.png')

Image('descriptionWordcloud.png')
# endregion

#   - #### Last review Wordcloud

# region
wholeReviewText = ''
for reviewText in trainCsv['LAST_REVIEW']:
    wholeReviewText = wholeReviewText + ' ' + reviewText

wc = WordCloud(width=600, height=600, background_color='white', stopwords=ENGLISH_STOP_WORDS)

wc.generate(wholeReviewText)
wc.to_file('lastReviewWordcloud.png')

Image('lastReviewWordcloud.png')
# endregion
# - ### *2 Extra Custom Questions*

#   - #### How many of the hosts have verified their identity

# region
identityIdentifiedPercentageSeries = trainCsv['HOST_IDENTITY_VERIFIED'].value_counts(normalize=True)

labels = []
sizes = []

for index, value in identityIdentifiedPercentageSeries.items():
    labels.append(index) 
    sizes.append(value)

labelDict = {
  "t": "Identified",
  "f": "Unidentified"
}

# Change the values of labels list
labels = [labelDict[x] for x in labels]

# Pie chart

# only "explode" the 2nd slice
explode = (0, 0.1)  

#add colors
colors = ['#DF0000','#00AD00']

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
 
plt.tight_layout()

plt.show()
# endregion
# ## __Recommendation System__

# We have already removed some stop words for the description wordcloud. Let's remove some punctuation and 
# decrease the number of words to be shown in the wordcloud.

# region
wholeDescriptionText = ''
for descriptionText in trainCsv['DESCRIPTION']:   

    # Remove any punctuation from the text
    for c in punctuation:
        descriptionText = descriptionText.replace(c, ' ')

    wholeDescriptionText = wholeDescriptionText + ' ' + descriptionText

wc = WordCloud(width=600, height=600, background_color='white', max_words=50, stopwords=ENGLISH_STOP_WORDS)

wc.generate(wholeDescriptionText)
wc.to_file('descriptionWordcloud2.png')

Image('descriptionWordcloud2.png')
# endregion

# We have already replaced nan values on name and description columns with "" (null string). So we are ready
# to copy the columns ID, NAME and DESCRIPTION.

# region
recommendCsv = trainCsv[['ID', 'NAME', 'DESCRIPTION']].copy()

# let's drop duplicate rows
recommendCsv.drop_duplicates(subset='ID', inplace=True)

# reset index of dataframe as we have dropped some rows
recommendCsv.reset_index(inplace=True, drop=True)

# make new column with the concatenation of name and description
recommendCsv['CONCATENATION'] = recommendCsv['NAME'] + recommendCsv['DESCRIPTION']
# endregion

#   - #### TF-IDF matrix of unigrams and bigrams for the CONCATENATION column

# region
tfIdfVectorizer = TfidfVectorizer(
                                max_features=None,
                                stop_words=ENGLISH_STOP_WORDS,
                                ngram_range=(1, 2)        # as we want both unigrams and bigrams
                                )


biUniGramsMatrix = tfIdfVectorizer.fit_transform(recommendCsv['CONCATENATION'])
# endregion

#   - #### Cosine Similarity

calculatedCosine = cosine_similarity(biUniGramsMatrix)

# The calculatedCosine is a 2d matrix which containsthe pairwise similarities between
# all samples in recommendCsv['CONCATENATION'] . For example at position (0,1) we have
# the cosine similarity score between id0 and id1. That means the similarity
# of every sentence/document to itself is 1 (hence the diagonal of the matrix will be all ones).

print('Pairwise dense output:\n {}\n'.format(calculatedCosine))

# Let's make a sparse matrix of cosine similarities.

calculatedCosineSparse = cosine_similarity(biUniGramsMatrix, dense_output=False)
print('Pairwise sparse output:\n {}\n'.format(calculatedCosineSparse))

# As we said the diagonal of the matrix will be all ones, so we will fill it with 0 so as to "skip" it
# when we will search for largerst values. We will also fill with 0 the matrix below diagonal because
# values are the same with those above diagonal. For example (10,2) and (2,10) have the same value.

# region
calculatedCosineCopy =  np.triu(calculatedCosine, 1)

print('Pairwise dense output:\n {}\n'.format(calculatedCosineCopy))
# endregion

def largestIndices(array, n):
    """Returns the n largest indices from a numpy array."""
    # flatten the array
    flatArray = array.flatten()
    indices = np.argpartition(flatArray, -n)[-n:]
    indices = indices[np.argsort(-flatArray[indices])]
    return np.unravel_index(indices, array.shape)

# region
mostSimilar100entries = largestIndices(calculatedCosineCopy, 100)

mostSimilar100entries
# endregion

# The above variable mostSimilar100entries is a tuple which consists of 2 arrays. The first array contains
# the indices of the first entry in each pair and the second array contains the indices of the second entry
# in each pair.

# So let's construct the dictionary of the 100 most similar pairs.

# region
mostSimilarDict = dict()
pairIndex = 1

for (firstEntry,secondEntry) in zip(*mostSimilar100entries):
    # keyString is of form pair_1, pair_2 etc
    keyString = "pair_" + str(pairIndex)
    # recommendCsv['ID'][firstEntry] is the ID for the first entry
    # recommendCsv['ID'][secondEntry] is the ID for the second entry
    # calculatedCosineCopy[firstEntry][secondEntry] is the cosine similarity score between these 2 entries
    mostSimilarDict[keyString] = (recommendCsv['ID'][firstEntry],
                                  recommendCsv['ID'][secondEntry],
                                  calculatedCosineCopy[firstEntry][secondEntry])
    pairIndex += 1 
    
mostSimilarDict
# endregion

# Όπως παρατηρούμε πολλά ζευγάρια έχουν score 1.0 παρόλο που είναι διαφορετικά τα IDS (δηλαδή δεν πρόκειται για το ίδιο ακίνητο)
# αλλά πρόκειται ίσως για συγκρότημα ακινήτων ή κάποιο hotel/hostel, οπότε ο ιδιοκτήτης έχει βάλλει το ίδιο όνομα και περιγραφή. 
# Επομένως είναι λογικό αυτό που παρατηρούμε.
