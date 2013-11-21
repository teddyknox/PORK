#! /usr/bin/env python

import requests
from peewee import *

db = SqliteDatabase('reddit.db')

class Post(Model):
    pid = PrimaryKeyField()
    fullname = CharField(max_length=10)
    title = CharField()
    score = IntegerField()
    created_utc = IntegerField()

    class Meta:
        database = db # this model uses the reddit database

# create the sqlite schema by uncommenting the next line the first time you run this
# Post.create_table()

if __name__ == '__main__':

    running = True
    request_params = {
        "limit": 100
    }

    while running:
        running = False
        response = requests.get('http://reddit.com/r/pics/new.json', params=request_params).json()
        request_params['before'] = response['data']['children'][-1]['data']['name']
        print request_params
        for post_json in response['data']['children']:
            post_json = post_json['data']
            post_dict = {
                'name': post_json['name'],
                'title': post_json['title'],
                'score': post_json['score'],
                'created_utc': post_json['created_utc']
            }
            print post_dict
            # post_obj = Post(**post_dict).save()

