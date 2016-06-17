import Scruffy.Game
import Scruffy.Main
import time

GLOBAL_MAX_VALUE = 0
SCORE = 0
ACTION_TAKEN = None
CUR_STATE = None


def main():
    Scruffy.Main.initialize()
    while True:
        step()
        time.sleep(0.1)


def restart():
    Scruffy.Main.restart(CUR_STATE, CUR_STATE, -GLOBAL_MAX_VALUE, ACTION_TAKEN, SCORE)
    Scruffy.Game.restart()
    global GLOBAL_MAX_VALUE, SCORE, ACTION_TAKEN, CUR_STATE
    GLOBAL_MAX_VALUE = 0
    SCORE = 0
    ACTION_TAKEN = None
    CUR_STATE = None


def step():
    if Scruffy.Game.game_over():
        restart()
        return
    global CUR_STATE, ACTION_TAKEN, GLOBAL_MAX_VALUE

    state = dict()
    max_val = 0
    global SCORE
    SCORE = 0
    gameboard = Scruffy.Game.get_gameboard()
    for i in range(len(gameboard)):
        for j in range(len(gameboard[i])):
            name = str(i) + "_" + str(j)
            state[name] = gameboard[i][j] if gameboard[i][j] is not None else 0
            if state[name] > max_val:
                max_val = state[name]
            SCORE += state[name]

    if CUR_STATE is not None:
        Scruffy.Main.reward_update(CUR_STATE, max_val, state, ACTION_TAKEN)

    action = Scruffy.Main.get_next_action(state, Scruffy.Game.get_illegal_actions())

    CUR_STATE = state
    ACTION_TAKEN = action
    GLOBAL_MAX_VALUE = max_val

    Scruffy.Game.do_action(action)

if __name__ == "__main__":
    main()