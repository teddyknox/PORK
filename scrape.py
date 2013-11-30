#! /usr/bin/env python

import requests
from peewee import *
from datetime import datetime, timedelta
import time
import argparse
import csv

# database
db = SqliteDatabase('reddit.db')

class Post(Model):
    pid = PrimaryKeyField()
    fullname = CharField(max_length=10)
    title = CharField()
    score = IntegerField()
    created_utc = IntegerField()

#     # image_id = IntegerField()
#     # unixtime = IntegerField()
#     # rawtime = CharField()
#     # title = CharField()
#     # total_votes = IntegerField()
#     # reddit_id = IntegerField()
#     # number_of_upvotes = IntegerField()
#     # subreddit = CharField()
#     # number_of_downvotes = IntegerField()
#     # localtime = IntegerField()
#     # score = IntegerField()
#     # number_of_comments = IntegerField()
#     # username = CharField()

    # class Meta:
    #     database = db # this model uses the reddit database

def load(filename):
    db.connect()
    Post.create_table(True) # Fail silently if table already exists
    with open(filename, 'r') as csvfile:
        for row in csv.DictReader(csvfile):
            print "%d\t%s" % (int(row['score']), row['title'])

def scrape():
    db.connect()
    Post.create_table(True) # Fail silently if table already exists

    running = True

    # unix timestamp of two weeks ago
    # threshold_timestamp = int((datetime.utcnow() - timedelta(weeks=2)).strftime("%s"))

    # get the fullname of the oldest submission saved
    # last_inserted = Post.select(Post.created_utc).limit(1).order_by(Post.created_utc.asc()).first()

    request_params = {
        "limit": 100
    }

    page = 1
    while running:
        # running = False
        page += 1
        print "page %d" % page
        start_time = int(datetime.now().strftime("%s"))
        # get submissions
        response = requests.get('http://reddit.com/r/pics/new.json', params=request_params).json()

        # get fullname of earliest post
        request_params['before'] = response['data']['children'][-1]['data']['name']

        # save each post
        for post_json in response['data']['children']:
            post_json = post_json['data']
            post_dict = {
                'fullname': post_json['name'],
                'title': post_json['title'],
                'score': post_json['score'],
                'created_utc': post_json['created_utc']
            }
            post_obj = Post(**post_dict).save()

        # should we wait until our next query? how long?
        stop_time = int(datetime.now().strftime("%s"))
        time_diff = stop_time - start_time
        if time_diff < 2:
            time.sleep(2 - time_diff)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrape reddit links using API.')
    parser.add_argument('action', help="Right now the only action is 'start'")
    args = parser.parse_args()
    if args.action == 'scrape':
        scrape()
    elif args.action == 'load':
        load('reddit.csv')