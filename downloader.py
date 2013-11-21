#! /usr/bin/env python

import requests
from peewee import *
from datetime import datetime, timedelta
import time
db = SqliteDatabase('reddit.db')

class Post(Model):
    pid = PrimaryKeyField()
    fullname = CharField(max_length=10)
    title = CharField()
    score = IntegerField()
    created_utc = IntegerField()

    class Meta:
        database = db # this model uses the reddit database

# uncomment this line the first time you run this to create the schema
# Post.create_table()

if __name__ == '__main__':

    running = True

    # unix timestamp of two weeks ago
    threshold_timestamp = int((datetime.utcnow() - timedelta(weeks=2)).strftime("%s"))

    # get the fullname of the oldest submission saved
    # last_inserted = Post.select(Post.created_utc).limit(1).order_by(Post.created_utc.asc()).first()
    # print type(last_inserted)
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
                'name': post_json['name'],
                'title': post_json['title'],
                'score': post_json['score'],
                'created_utc': post_json['created_utc']
            }
            if post_dict['created_utc'] < threshold_timestamp:
                print "before threshold"
                post_obj = Post(**post_dict).save()

        # should we wait until our next query? how long?
        stop_time = int(datetime.now().strftime("%s"))
        time_diff = stop_time - start_time
        if time_diff < 2:
            time.sleep(2 - time_diff)