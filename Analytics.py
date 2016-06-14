from pymongo import MongoClient
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
    return json.dumps([{"time": x["time"], "score": x["reward"]} for x in scores.find()]), 201


