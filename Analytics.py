
import pymongo
import json
import numpy as np

import Database


def get_stats(agent):
    return json.dumps({"max": Database.get_high_score(agent.name),
                       "count": Database.scores_count(agent.name)})


def get_fitted_line(agent):
    y = list()
    x = list()
    for i in Database.get_scores(agent.name).sort('time', pymongo.ASCENDING):
        y.append(i["reward"])
        x.append(i["time"])

    coefs = np.polyfit(x, y, 1)
    fit = lambda z: z*coefs[0] + coefs[-1]
    return json.dumps([(x[0], fit(x[0])), (x[-1], fit(x[-1]))])


def get_reward_data(agent):
    return json.dumps(window_averages(Database.get_scores(agent.name).sort('time', pymongo.ASCENDING))), 201


def window_averages(data, window=None):
    if window is None:
        window = max(int((data.count()/50)), 1)
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
