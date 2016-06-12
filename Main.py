from flask import Flask, request, Response
import json
import random
from pymongo import MongoClient

app = Flask(__name__)

# Up Down Left Right
ACTIONS = [0, 1, 2, 3]
GRID_SIZE = 4
ALPHA = 0.1
GAMMA = 0.9
Exploration = 0.05

client = None
active_count = 0
game_id = 0


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


@app.route("/")
def home():
    return "<h1><a href=\"/api\">This ain't a website yo</a></h1>"


@app.route("/api")
def api_home():
    return "<ul><li>/api/initialize</li><li>/api/next_action</li><li>/api/reward_update</li></ul>"


@app.route("/api/initialize", methods=['POST'])
def initialize():
    global client, active_count, game_id
    if client is None:
        client = MongoClient()
        active_count += 1
        random.seed()
        game_id += 1
    return (json.dumps({"game_id": game_id}), 201) if client is not None else (json.dumps("Error in client setup"), 501)


@app.route("/api/get_action", methods=['POST'])
def get_next_action():
    if client is None:
        initialize()
    state = request.json["state"]
    illegals = request.json["illegals"]
    if state is None:
        return json.dumps(random.choice(ACTIONS))
    database = client["AI2048"]
    states = database.states
    record = states.find_one({"state": state})
    if record is None:
        record = create_new_entry(state, states)
    return json.dumps({"action": get_e_greedy_action(record["actions"], illegals)}), 201


@app.route("/api/reward_update", methods=['POST'])
def update_reward():
    state = request.json["state"]
    next_state = request.json["next_state"]
    reward = request.json["reward"]
    action_taken = request.json["action_taken"]
    if state is None:
        return json.dumps("Reward update is not acceptable"), 501
    reward_update(state, float(reward), next_state, action_taken)
    return json.dumps({"game_id": game_id}), 201


def reward_update(state, action_reward, next_state, action_taken):
    database = client["AI2048"]
    states = database.states
    record = states.find_one({"state": state})
    if record is None:
        return json.dumps("Reward update state doesn't exist????"), 501

    next_record = states.find_one({"state": next_state})
    if next_record is None:
        next_record = create_new_entry(next_state, states)

    reward = action_reward
    reward += ALPHA*(GAMMA * next_record["actions"][str(max(next_record["actions"], key=next_record["actions"].get))] -
                    record["actions"][action_taken])
    record["actions"][action_taken] += reward
    states.update({'_id': record['_id']}, record)


@app.route("/api/restart", methods=['POST'])
def restart():
    state = request.json["state"]
    next_state = request.json["next_state"]
    reward = request.json["reward"]
    action_taken = request.json["action_taken"]
    if state is None:
        return json.dumps("Reward update is not acceptable"), 501
    reward_update(state, float(reward), next_state, action_taken)
    return json.dumps({"game_id": game_id}), 201


@app.route("/api/get_script", methods=["GET"])
def get_script():
    with open('script.js', 'r') as myfile:
        data = myfile.read()
    return Response(data, mimetype='application/javascript')


def get_e_greedy_action(actions, illegals):
    illegals = [str(x) for x in illegals]
    if random.uniform(0, 1) > Exploration:
        max_value = next(iter(actions.values()))
        keys = list()
        for key, val in actions.items():
            if val > max_value:
                keys = [key]
                max_value = val
            elif val == max_value:
                keys.append(key)
    else:
        keys = list(actions.keys())
    keys = [x for x in keys if x not in illegals]
    if len(keys) == 0:
        keys = [x for x in list(actions.keys()) if x not in illegals]
    return random.choice(keys)


def create_new_entry(state, collection):
    new_entry = dict()
    new_entry["state"] = state
    new_entry["actions"] = dict()
    for i in ACTIONS:
        new_entry["actions"][str(i)] = random.gauss(0, 1)
    collection.insert_one(new_entry)
    return new_entry


if __name__ == "__main__":
    app.run(host="0.0.0.0")
