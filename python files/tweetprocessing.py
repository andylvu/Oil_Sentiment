import pandas as pd
import re
from textblob import TextBlob

# read the csv file of tweets
df = pd.read_csv('out1.csv', index_col='Date')
df2 = pd.read_csv('CrudeWTIDaily.csv', index_col='Date')


# edit the column to get rid of specific time to only keep date
df.index = df.index.str.slice(0, 10)
# print(df.head(5))


# function to clean the tweets
def cleantweet(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # remove @mentions
    text = re.sub(r'#', '', text)  # remove the '#' symbol
    text = re.sub(r'RT[\s]+', '', text)  # remove RT
    text = re.sub(r'https?:\/\/\S+', '', text)  # remove hyperlink
    return text


# apply tweet cleaning to tweets
df['Text'] = df['Text'].apply(cleantweet)
print('clean tweets')
# print(df.head(5))


# function to get polarity
def getpolarity(text):
    return TextBlob(text).sentiment.polarity


# apply polarity
df['Polarity'] = df['Text'].apply(getpolarity)
print('get polarity')
print(df.head(5))

# write polarity to csv for observations
# df.to_csv('analyzed_tweets.csv')

# count the amount of negative polarity
count = df['Polarity'].lt(0).sum()
print(count, 'count of negative tweets')

# count of tweets with either NaN or 0 polarity
count1 = df['Polarity'].eq(0).sum()
print(count1, 'count of 0 polarity tweets')

# turns date string into datetime
df.index = pd.to_datetime(df.index)

# convert polarity string to float
df['Polarity'] = df.Polarity.astype(float)

# calculating the average polarity by day
df_average_date = df.Polarity.resample('D').mean()

# calculate twitter volume for daily
df_tweet_volume = df.index.value_counts()

# new dataframe with date, polarity and volume
df1 = pd.DataFrame(df_average_date)
df1['Volume'] = df_tweet_volume
print(df1)

# convert oil dataframe indext to datetime
df2.index = pd.to_datetime(df2.index)

# merge dataframes of oil prices and tweets
merged_data = pd.merge(df1, df2, how='outer', on='Date')
print(merged_data)

# drop date values that have no information due to non trading day
merged_data = merged_data[merged_data['Price'].notna()]

# replace NaN values with zeroes
merged_data.fillna(0, inplace=True)
print(merged_data)

# output final dataframe to csv
merged_data.to_csv('oil_final1.csv')
