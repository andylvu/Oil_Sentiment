import pandas as pd
import tweepy
import datetime
import time
from datetime import timedelta

class tweetpull:
    # initialize tweepy client using bearer token
    def __init__(self, bearer_token):
        self.client = tweepy.Client(bearer_token = bearer_token)

    def collect_tweets(self, query, start_date, end_date, days = 3, sleep_interval = 1.1):
        tweets = []

        while start_date < datetime.now():
            tweet_data = self._fetch_tweets(query, start_date, end_date)
            tweets.extend(tweet_data)
            start_date = end_date
            end_date += timedelta(days=1)
            time.sleep(sleep_interval)


            
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
