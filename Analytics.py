from pymongo import MongoClient
import pymongo
import json
import numpy as np

client = None


def initialize():
    global client
    client = MongoClient()


def get_stats():
    if client is None:
        initialize()
    database = client["AI2048"]

    return json.dumps({"max": database.neural_scores.find_one(sort=[("reward", pymongo.DESCENDING)])["reward"],
                       "count": database.neural_scores.count()})


def get_fitted_line():
    if client is None:
        initialize()
    database = client["AI2048"]
    y = list()
    x = list()
    for i in database.neural_scores.find().sort('time', pymongo.ASCENDING):
        y.append(i["reward"])
        x.append(i["time"])

    coefs = np.polyfit(x, y, 1)
    fit = lambda x: x*coefs[0] + coefs[-1]
    return json.dumps([(x[0], fit(x[0])), (x[-1], fit(x[-1]))])


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
        window = max(int((data.count()/100)), 1)
    ret_list = list()
    cnt = 0
    avg = 0
    last = None
    for i in data:
        avg += i["reward"]
        last = i
        cnt += 1
        if cnt == window:
            ret_list.append({"time": i["time"], "score": float(avg)/cnt})
            last = None
            avg = 0
            cnt = 0
    if last is not None:
        ret_list.append({"time": last["time"], "score": float(avg)/cnt})
    return ret_list
