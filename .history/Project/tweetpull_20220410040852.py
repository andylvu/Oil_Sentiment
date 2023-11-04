import pandas as pd
import tweepy
import datetime
import time
from datetime import timedelta

# Authentication with tokens
consumer_key = '1Zaw8qUiBgFPGTVN0WAGOsGAW'
consumer_secret = '68Cr7am4XHR0rNH2jkJQFs0yBEJtRJt307tEviSv18H3pGXlbm'
access_token = '1498445144537931778-dlmLDiog3Zzyli1Eo0gd3IuUvOgBgD'
access_token_secret = '6YmevHwhVCJGz4Yw5nVt4HfZnoJE4sHntr96o38flxVht'


# tweepy client using academic research token
client = tweepy.Client(bearer_token='AAAAAAAAAAAAAAAAAAAAAHwkbQEAAAAAU%2Ffmq5LdPnQvnS%2BBKxxP5WzikUA%3DGEHkRGtbIeZFnDfyRFOFjLvcOcIrgXOOwKmExeqArJZ90bgKHj')


# query with start and end dates
query = 'oil wti gas -is:retweet'
endtime = datetime.datetime(2010, 1, 1)

# days to iterate
days = 3

# request count
request = 0

# list to for pulled tweets to be appended to
tweet_text = []
tweet_date = []

# api request count
request_count = 0

# for loop to iterate through the days
for number in range(days):
    # api to pull tweets
    response = client.search_all_tweets(query=query, max_results=500, tweet_fields=['created_at'], end_time=endtime)
    for tweet in response.data:
        tweet_date.append(tweet.created_at)
        tweet_text.append(tweet.text)
    endtime += timedelta(days=1)
    print(endtime)
    time.sleep(1.1)
    request += 1
    if request % 300 == 0:
        print('waiting for 15 minute window')
        time.sleep(901)

# putting data into a dataframe
df = pd.DataFrame(tweet_date, columns=['Date'])
df['Text'] = tweet_text

print(df)

df.to_csv('out.csv', index=False)
