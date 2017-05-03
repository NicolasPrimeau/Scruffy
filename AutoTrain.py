import Database
from Game import Game
import datetime
import sys
import os
import signal

import warnings

from agents.ImaginativeNNAgent import ImaginativeNNAgent

warnings.filterwarnings("ignore")

GLOBAL_MAX_VALUE = 0
SCORE = 0
CUR_STATE = None
MAX_SCORE = 0
GAMES = 0
REWARD = 0

# Up right down Left
ACTIONS = (0, 1, 2, 3)
GRID_SIZE = 4
ALPHA = 0.1
GAMMA = 0.9
Exploration = 0.05
WRONG_MOVES = 0

main_agent_type = ImaginativeNNAgent

AGENT = None
SAVE_STEP = 100
LIMITER = None


def main(agent_type, no_print=False):
    if no_print:
        sys.stdout = open(os.devnull, 'w')
    global MAX_SCORE, LIMITER, GRID_SIZE, AGENT
    game = Game()
    agent = agent_type(name="BasicNet", actions=ACTIONS, features=GRID_SIZE**2,
    # agent = agent_type(actions=ACTIONS, features=GRID_SIZE ** 2,
                       elligibility_trace=True, game_size=GRID_SIZE, forgetting_factor=0.5)
    AGENT = agent
    game.restart()
    MAX_SCORE = Database.get_high_score(agent.name)
    print(str(datetime.datetime.now()) + " Starting up")
    while LIMITER is None or GAMES < LIMITER:
        step(game, agent)
    agent.save()
    return 1


def restart(game, agent):
    global CUR_STATE, GAMES, SAVE_STEP, GLOBAL_MAX_VALUE, SCORE, REWARD, WRONG_MOVES
    print(str(datetime.datetime.now()) + " Still Alive, Game: " + str(GAMES) + ", High Score: " + str(MAX_SCORE) +
          ", Max Value: " + str(GLOBAL_MAX_VALUE) + ", Score: " + str(SCORE) + ", Reward: " + str(REWARD) +
          ", Wrong Moves: " + str(WRONG_MOVES))
    agent.learn()
    Database.save_score(agent.name, SCORE)
    game.restart()
    CUR_STATE = None
    GLOBAL_MAX_VALUE = 0
    SCORE = 0
    GAMES += 1
    REWARD = 0
    WRONG_MOVES = 0
    if GAMES % SAVE_STEP == 0:
        print("Saving Agent State")
        agent.save()


def step(game, agent):
    global CUR_STATE, GLOBAL_MAX_VALUE, SCORE, MAX_SCORE, REWARD, WRONG_MOVES, ACTION_Q
    if CUR_STATE is None:
        GLOBAL_MAX_VALUE = 0
        CUR_STATE, SCORE = game.get_state()

    illegals = game.get_illegal_actions()
    action = agent.get_action(CUR_STATE)

    while action in illegals:
        agent.give_reward(-2048)
        WRONG_MOVES += 1
        action = agent.get_action(CUR_STATE)

    merged_val = game.do_action(action)

    if game.game_over():
        print("Game Over")
        agent.give_reward(-GLOBAL_MAX_VALUE)
        restart(game, agent)
    else:
        CUR_STATE, SCORE = game.get_state()
        agent.give_reward(merged_val)
        REWARD += merged_val
        max_val = max(CUR_STATE, key=(lambda key: CUR_STATE[key]))
        if GLOBAL_MAX_VALUE < CUR_STATE[max_val]:
            GLOBAL_MAX_VALUE = CUR_STATE[max_val]
        if SCORE > MAX_SCORE:
            MAX_SCORE = SCORE


def exit_handler(sig, frame):
    global AGENT
    print("Sig int received, saving agent")
    AGENT.save()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, exit_handler)
    main(main_agent_type)
