from flask import Flask, request, Response, render_template
import json
import random
import datetime
import Analytics
from pybrain.rl.learners.valuebased import NFQ
from pybrain.rl.learners.valuebased import ActionValueNetwork
from pybrain.rl.agents import LearningAgent
from pymongo import MongoClient
import pickle
from bson.binary import Binary

app = Flask(__name__)

# Up right down Left
ACTIONS = [0, 1, 2, 3]
GRID_SIZE = 4
ALPHA = 0.1
GAMMA = 0.9
Exploration = 0.05

GAME_BOARD_LENGTH = 4

client = None
game_id = 0

agent = None


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


@app.route("/")
def home():
    return "<h1>Whatup</h1>"


@app.route("/analytics")
def analytics():
    return render_template('analytics.html')


@app.route("/analytics/get_scores", methods=['GET'])
def get_reward_data():
    return Analytics.get_reward_neural_data()


@app.route("/api")
def api_home():
    return "<ul><li>/api/get_script</li><li>/api/initialize</li><li>/api/restart</li>" \
           "<li>/api/next_action</li><li>/api/reward_update</li></ul>"


@app.route("/api/initialize", methods=['POST'])
def initialize():
    global agent, client, game_id
    if client is None:
        client = MongoClient()
        random.seed()
        game_id += 1
    if agent is None:
        initialize_network()
    return (json.dumps({"game_id": game_id}), 201) if client is not None else (json.dumps("Error in client setup"), 501)


def initialize_network():
    global agent
    if agent is None:
        record = client["AI2048"].networks.find_one()
        if record is None:
            controller = ActionValueNetwork(GAME_BOARD_LENGTH * GAME_BOARD_LENGTH, len(ACTIONS))
        else:
            controller = ActionValueNetwork(GAME_BOARD_LENGTH * GAME_BOARD_LENGTH, len(ACTIONS))
            controller.network = pickle.loads(record["data"])
            controller.network.sorted = False
            controller.network.sortModules()
        agent = LearningAgent(controller, NFQ())
    agent.newEpisode()


@app.route("/api/get_action", methods=['POST'])
def get_next_action_handler():
    state = request.json["state"]
    return json.dumps({"action": get_next_action(state)}), 201


@app.route("/api/reward_update", methods=['POST'])
def update_reward_handler():
    if agent or client is None:
        return
    state = request.json["state"]
    reward = request.json["reward"]

    if state is None:
        return json.dumps("Reward update is not acceptable"), 501
    reward_update(float(reward))
    return json.dumps({"game_id": game_id}), 201


def reward_update(action_reward):
    global agent
    agent.giveReward(action_reward)


@app.route("/api/restart", methods=['POST'])
def restart_handler():
    reward = request.json["reward"]
    score = request.json["score"]
    restart(reward, score)
    return json.dumps({"game_id": game_id}), 201


def restart(reward, score):
    global agent
    client["AI2048"].neural_scores.insert_one({"reward": score, "time": datetime.datetime.now().timestamp()})
    reward_update(float(reward))
    agent.newEpisode()
    agent.learn()
    save_agent()


def save_agent():
    global agent, client
    if agent is None or client is None:
        return
    client["AI2048"].networks.update_one({"network": 1},
                                         {"$set": {"data": Binary(pickle.dumps(agent.module.network))}
                                          },
                                         upsert=True)


@app.route("/api/get_script", methods=["GET"])
def get_script():
    with open('script.js', 'r') as myfile:
        data = myfile.read()
    return Response(data, mimetype='application/javascript')


@app.route("/static/analytics.js", methods=["GET"])
def get_analytics():
    with open('static/analytics.js', 'r') as myfile:
        data = myfile.read()
    return Response(data, mimetype='application/javascript')


def get_next_action(state):
    global agent
    if state is None:
        return random.choice(ACTIONS)
    agent.integrateObservation(map_state_to_inputs(state))
    if agent is None:
        initialize_network()

    return str(ACTIONS[int(agent.getAction())])


def map_state_to_inputs(state):
    state_mapping = list()
    for i in range(GAME_BOARD_LENGTH):
        for j in range(GAME_BOARD_LENGTH):
            key = str(i) + "_" + str(j)
            state_mapping.append(state[key])
    return state_mapping


if __name__ == "__main__":
    app.run(host="0.0.0.0", threaded=True)
