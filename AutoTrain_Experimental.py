import Game
import Main_Experimental
import datetime

GLOBAL_MAX_VALUE = 0
SCORE = 0
CUR_STATE = None
MAX_SCORE = 0
GAMES = 0
REWARD = 0
STEP_TIME = 0.1


def main():
    global STEP_TIME, MAX_SCORE
    Main_Experimental.initialize()
    Game.restart()
    MAX_SCORE = Main_Experimental.get_high_score()
    print(str(datetime.datetime.now()) + " Starting up")
    while True:
        step()


def restart():
    global CUR_STATE, GAMES, GLOBAL_MAX_VALUE, SCORE, REWARD
    print(str(datetime.datetime.now()) + " Still Alive, Game: " + str(GAMES) + ", High Score: " + str(MAX_SCORE) +
          ", Max Value: " + str(GLOBAL_MAX_VALUE) + ", Score: " + str(SCORE) + ", Reward: " + str(REWARD))
    print("gob max:" + str(GLOBAL_MAX_VALUE))
    Main_Experimental.restart(-GLOBAL_MAX_VALUE, SCORE)
    Game.restart()
    CUR_STATE = None
    GLOBAL_MAX_VALUE = 0
    SCORE = 0
    GAMES += 1
    REWARD = 0


def step():
    global CUR_STATE, GLOBAL_MAX_VALUE, SCORE, MAX_SCORE, REWARD

    if CUR_STATE is None:
        GLOBAL_MAX_VALUE = 0
        CUR_STATE, SCORE = get_state()

    illegals = Game.get_illegal_actions()
    action = Main_Experimental.get_next_action(CUR_STATE)
    while action in illegals:
        Main_Experimental.reward_update(-1000)
        REWARD += -1000
        action = Main_Experimental.get_next_action(CUR_STATE)

    merged_val = Game.do_action(action)

    if Game.game_over():
        print("Game Over")
        restart()
    else:
        CUR_STATE, SCORE = get_state()
        Main_Experimental.reward_update(merged_val)
        print(merged_val)
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
    main()
