import Game
import Main
import time

GLOBAL_MAX_VALUE = 0
SCORE = 0
ACTION_TAKEN = None
CUR_STATE = None

STEP_TIME = 0.1

def main():
    global STEP_TIME
    Main.initialize()
    cnter = 0
    while True:
        step()
        cnter +=1
        if cnter > 10:
           cnter = 0
           print("Still Alive")

def restart():
    global GLOBAL_MAX_VALUE, SCORE, ACTION_TAKEN, CUR_STATE
    Main.restart(CUR_STATE, CUR_STATE, -GLOBAL_MAX_VALUE, ACTION_TAKEN, SCORE)
    Game.restart()    
    GLOBAL_MAX_VALUE = 0
    SCORE = 0
    ACTION_TAKEN = None
    CUR_STATE = None


def step():
    if Game.game_over():
        restart()
        return
    global CUR_STATE, ACTION_TAKEN, GLOBAL_MAX_VALUE

    state = dict()
    max_val = 0
    global SCORE
    SCORE = 0
    gameboard = Game.get_gameboard()
    for i in range(len(gameboard)):
        for j in range(len(gameboard[i])):
            name = str(i) + "_" + str(j)
            state[name] = gameboard[i][j] if gameboard[i][j] is not None else 0
            if state[name] > max_val:
                max_val = state[name]
            SCORE += state[name]

    if CUR_STATE is not None:
        Main.reward_update(CUR_STATE, max_val, state, ACTION_TAKEN)

    action = Main.get_next_action(state, Game.get_illegal_actions())

    CUR_STATE = state
    ACTION_TAKEN = action
    GLOBAL_MAX_VALUE = max_val

    Game.do_action(action)

if __name__ == "__main__":
    main()
