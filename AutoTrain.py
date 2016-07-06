import Database
import Game
import datetime
import sys
import os

import warnings

from agents.ClusterAgent import ClusterAgent
from agents.DiscreteAgent import DiscreteAgent
from agents.DiscreteGraphAgent import DiscreteGraphAgent
from agents.DiscreteNeighbourAgent import DiscreteNeighbourAgent
from agents.DiscreteTreeAgent import DiscreteTreeAgent
from agents.NeuralNetAgent import NeuralNetAgent
from agents.DiscreteStateLookupAgent import DiscreteStateLookupAgent
from agents.TensorFlowAgent import TensorFlowAgent

warnings.filterwarnings("ignore")

GLOBAL_MAX_VALUE = 0
SCORE = 0
CUR_STATE = None
MAX_SCORE = 0
GAMES = 0
REWARD = 0

# Up right down Left
ACTIONS = [0, 1, 2, 3]
GRID_SIZE = 4
ALPHA = 0.1
GAMMA = 0.9
Exploration = 0.05
WRONG_MOVES = 0

main_agent_type = TensorFlowAgent
SAVE_STEP = 1000
LIMITER = None


def main(agent_type, no_print=False):
    if no_print:
        sys.stdout = open(os.devnull, 'w')
    global MAX_SCORE, LIMITER, GRID_SIZE
    agent = agent_type(actions=ACTIONS, features=GRID_SIZE**2, alpha=ALPHA, gamma=GAMMA, exploration=Exploration,
                       elligibility_trace=True, game_size=GRID_SIZE, forgetting_factor=0.5)
    Game.restart()
    MAX_SCORE = Database.get_high_score(agent.name)
    print(str(datetime.datetime.now()) + " Starting up")
    while LIMITER is None or GAMES < LIMITER:
        step(agent)
    agent.save()
    return 1


def restart(agent):
    global CUR_STATE, GAMES, SAVE_STEP, GLOBAL_MAX_VALUE, SCORE, REWARD, WRONG_MOVES
    print(str(datetime.datetime.now()) + " Still Alive, Game: " + str(GAMES) + ", High Score: " + str(MAX_SCORE) +
          ", Max Value: " + str(GLOBAL_MAX_VALUE) + ", Score: " + str(SCORE) + ", Reward: " + str(REWARD) +
          ", Wrong Moves: " + str(WRONG_MOVES))
    agent.learn()
    Database.save_score(agent.name, SCORE)
    Game.restart()
    CUR_STATE = None
    GLOBAL_MAX_VALUE = 0
    SCORE = 0
    GAMES += 1
    REWARD = 0
    WRONG_MOVES = 0
    if GAMES % SAVE_STEP == 0:
        print("Saving Agent State")
        agent.save()


def step(agent):
    global CUR_STATE, GLOBAL_MAX_VALUE, SCORE, MAX_SCORE, REWARD, WRONG_MOVES

    if CUR_STATE is None:
        GLOBAL_MAX_VALUE = 0
        CUR_STATE, SCORE = get_state()

    illegals = Game.get_illegal_actions()
    action = agent.get_action(CUR_STATE)

    while action in illegals:
        agent.give_reward(-2048)
        WRONG_MOVES += 1
        action = agent.get_action(CUR_STATE)

    merged_val = Game.do_action(action)

    if Game.game_over():
        print("Game Over")
        agent.give_reward(-GLOBAL_MAX_VALUE)
        restart(agent)
    else:
        CUR_STATE, SCORE = get_state()
        agent.give_reward(merged_val)
        REWARD += merged_val
        if GLOBAL_MAX_VALUE < merged_val:
            GLOBAL_MAX_VALUE = merged_val
        if SCORE > MAX_SCORE:
            MAX_SCORE = SCORE


def get_state():
    state = dict()
    score = 0
    gameboard = Game.get_gameboard()
    for i in range(len(gameboard)):
        for j in range(len(gameboard[i])):
            name = str(i) + "_" + str(j)
            state[name] = gameboard[i][j] if gameboard[i][j] is not None else 0
            score += state[name]
    return state, score


if __name__ == "__main__":
    main(main_agent_type)
