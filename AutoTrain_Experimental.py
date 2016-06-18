import Game
import Main_Experimental
import datetime

GLOBAL_MAX_VALUE = 0
SCORE = 0
CUR_STATE = None
MAX_SCORE = 0

LOST = 0
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
    global CUR_STATE
    print(str(datetime.datetime.now()) + " Still Alive, High Score: " + str(MAX_SCORE) + ", Max Value: " +
          str(GLOBAL_MAX_VALUE) + ", Score: " + str(SCORE))
    Main_Experimental.restart(-GLOBAL_MAX_VALUE, SCORE)
    Game.restart()
    CUR_STATE = None


def step():
    global LOST, CUR_STATE, GLOBAL_MAX_VALUE, SCORE, MAX_SCORE

    if CUR_STATE is None:
        CUR_STATE, GLOBAL_MAX_VALUE, SCORE = get_state()

    illegals = Game.get_illegal_actions()
    action = Main_Experimental.get_next_action(CUR_STATE)
    while action in illegals:
        Main_Experimental.reward_update(-100000)
        action = Main_Experimental.get_next_action(CUR_STATE)
    Game.do_action(action)

    if Game.game_over():
        print("Game Over")
        LOST += 1
        restart()
        return

    CUR_STATE, max_val, SCORE = get_state()
    if GLOBAL_MAX_VALUE < max_val:
        GLOBAL_MAX_VALUE = max_val
        Main_Experimental.reward_update(GLOBAL_MAX_VALUE)
    else:
        Main_Experimental.reward_update(0)

    if SCORE > MAX_SCORE:
        MAX_SCORE = SCORE


def get_state():
    state = dict()
    max_val = 0
    score = 0
    gameboard = Game.get_gameboard()
    for i in range(len(gameboard)):
        for j in range(len(gameboard[i])):
            name = str(i) + "_" + str(j)
            state[name] = gameboard[i][j] if gameboard[i][j] is not None else 0
            if state[name] > max_val:
                max_val = state[name]
            score += state[name]
    return state, max_val, score


if __name__ == "__main__":
    main()

