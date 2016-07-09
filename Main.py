from flask import Flask, request, Response, render_template
import json
import random
import Analytics
from agents.Agent import map_state_to_inputs
from agents.AutoLookAheadTensorFlowAgent import AutoLookAheadTensorFlowAgent, translate_state_to_game_board

app = Flask(__name__)

# Up right down Left
ACTIONS = [0, 1, 2, 3]
GRID_SIZE = 4
ALPHA = 0.1
GAMMA = 0.9
Exploration = 0.05

AGENT_TYPE = AutoLookAheadTensorFlowAgent
AGENT = None
game_id = 0


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
    global AGENT
    return render_template('analytics.html')


@app.route("/analytics/get_scores", methods=['GET'])
def get_reward_data():
    global AGENT
    return Analytics.get_reward_data(AGENT)


@app.route("/analytics/get_fitted_line", methods=['GET'])
def get_line():
    global AGENT
    return Analytics.get_fitted_line(AGENT)


@app.route("/analytics/get_stats", methods=['GET'])
def get_stats():
    global AGENT
    return Analytics.get_stats(AGENT)


@app.route("/api")
def api_home():
    return "<ul><li>/api/get_script</li><li>/api/initialize</li><li>/api/restart</li>" \
           "<li>/api/next_action</li><li>/api/reward_update</li></ul>"


@app.route("/api/initialize", methods=['POST'])
def initialize():
    global game_id
    game_id += 1
    return json.dumps({"game_id": game_id}), 201


setting_up = False


def setup():
    global AGENT, Exploration, setting_up, ALPHA, GAMMA, ACTIONS, game_id, GRID_SIZE
    if not setting_up and AGENT is None:
        setting_up = True
        AGENT = AGENT_TYPE(actions=ACTIONS, features=GRID_SIZE**2, game_size=4, alpha=ALPHA, gamma=GAMMA,
                           exploration=Exploration, elligibility_trace=True)
        setting_up = False
    random.seed()


@app.route("/api/get_action", methods=['POST'])
def get_next_action_handler():
    global AGENT
    state = request.json["state"]
    illegals = request.json["illegals"]
    action = AGENT.get_action(state)
    illegals = [int(i) for i in illegals]
    action = AGENT.get_action(translate_state_to_game_board(map_state_to_inputs(state)))
    return json.dumps({"action": str(action)}), 200


@app.route("/api/reward_update", methods=['POST'])
def update_reward_handler():
    global AGENT
    reward = request.json["reward"]
    return json.dumps({"game_id": game_id}), 201


@app.route("/api/restart", methods=['POST'])
def restart_handler():
    reward = request.json["reward"]
    score = request.json["score"]
    global AGENT
    AGENT.clean()
    AGENT.load()
    return json.dumps({"game_id": game_id}), 201


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


if __name__ == "__main__":
    setup()
    app.run(host="0.0.0.0", threaded=True)
