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
    return json.dumps(window_averages(scores.find().sort('time', pymongo.ASCENDING))), 201


def get_reward_neural_data():
    if client is None:
        initialize()
    scores = client["AI2048"].neural_scores
    return json.dumps(window_averages(scores.find().sort('time', pymongo.ASCENDING))), 201


def window_averages(data, window=None):
    if window is None:
        window = max([int((data.count()/100)), 1])

    ret_list = list()
    cnt = 0
    avg = 0
    last = None
    for i in data:
        avg += i["reward"]
        cnt += 1
        if cnt == window:
            ret_list.append({"time": i["time"], "score": float(avg)/cnt})
            last = None
            avg = 0
            cnt = 0
        last = i
    if last is not None:
        ret_list.append({"time": last["time"], "score": float(avg)/cnt})
    return ret_list
