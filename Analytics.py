from pymongo import MongoClient
import pymongo
import json

client = None


def initialize():
    global client
    client = MongoClient()


def get_reward_data():
    if client is None:
        initialize()
    database = client["AI2048"]
    scores = database.scores
    return json.dumps([{"time": x["time"], "score": x["reward"]} for x in scores.find().sort('time', pymongo.ASCENDING)]), 201


def get_reward_neural_data():
    if client is None:
        initialize()
    scores = client["AI2048"].neural_scores
    return json.dumps([{"time": x["time"], "score": x["reward"]} for x in scores.find().sort('time', pymongo.ASCENDING)]), 201

