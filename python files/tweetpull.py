import pandas as pd
import tweepy
import datetime
import time
from datetime import timedelta

# rewritten code

class tweetpull:
    # initialize tweepy client using bearer token
    def __init__(self, bearer_token):
        self.client = tweepy.Client(bearer_token = bearer_token)

    def collect_tweets(self, query, start_date, end_date, days = 3, sleep_interval = 1.1, request_limit = 300):
        tweets = []

        while start_date < datetime.now():
            tweet_data = self.fetch_tweets(query, start_date, end_date)
            tweets.extend(tweet_data)
            start_date = end_date
            end_date += timedelta(days=1)
            time.sleep(sleep_interval)

            # check for request limit and wait 15 minutes
            self.wait_for_rate_limit(request_limit)

        return tweets
    
    def wait_for_rate_limit(self, request_limit):
        if self.client.rate_limit['/tweets/search/all'] <= request_limit:
            print('Reached rate limit. Waiting for 15 minutes...')
            time.sleep(901)
    
    def fetch_tweets(self, query, start_date, end_date):
        tweet_data = []

        response = self.client.search_all_tweets(query=query, max_results = 500, tweet_fields=['created_at'], end_time=end_date)
        for tweet in response.data:
            tweet_data.append({
                'Date': tweet.created_at,
                'Text': tweet.text
            })

        return tweet_data
    
    def collect_and_save_tweets(self, query, output_file, start_date, days = 3, sleep_interval=1.1):
        tweets = self.collect_tweets(query, start_date, start_date + timedelta(days=days), sleep_interval=sleep_interval)
        df = pd.DataFrame(tweets)
        df.to_csv(output_file, index=False)
        return df

if __name__ == "__main__":
    bearer_token = ''
    
    collector = tweetpull(bearer_token)
    
    query = 'oil wti gas -is:retweet'
    start_date = datetime(2010, 1, 1)
    output_file = 'out.csv'
    
    df = collector.collect_and_save_tweets(query, output_file, start_date)
    print(df)



# orginal code
"""
# tweepy client using academic research token
client = tweepy.Client(bearer_token='')


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
"""