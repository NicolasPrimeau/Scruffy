import random

GAME_BOARD_LENGTH = 4
GAME_BOARD = [[None for i in range(GAME_BOARD_LENGTH)] for j in range(GAME_BOARD_LENGTH)]

PROB_2 = 0.9

# Up right down Left
DIRECTIONS = [0, 1, 2, 3]


def do_action(direction):
    direction = int(direction)
    global GAME_BOARD
    if direction not in DIRECTIONS:
        return 0

    merged = [[False for j in range(GAME_BOARD_LENGTH)] for i in range(GAME_BOARD_LENGTH)]

    if direction == 0:
        for i in range(1, GAME_BOARD_LENGTH):
            for j in range(GAME_BOARD_LENGTH):
                if GAME_BOARD[i][j] is not None:
                    rows = i
                    merg = False
                    while rows > 0 and not merg and not merged[rows-1][j]:
                        if GAME_BOARD[rows-1][j] == GAME_BOARD[rows][j]:
                            GAME_BOARD[rows-1][j] *= 2
                            GAME_BOARD[rows][j] = None
                            merged[rows-1][j] = True
                            merg = True
                        elif GAME_BOARD[rows-1][j] is None:
                            GAME_BOARD[rows-1][j] = GAME_BOARD[rows][j]
                            GAME_BOARD[rows][j] = None
                        rows -= 1

    elif direction == 1:
        for j in reversed(range(0, GAME_BOARD_LENGTH-1)):
            for i in range(GAME_BOARD_LENGTH):
                if GAME_BOARD[i][j] is not None:
                    cols = j
                    merg = False
                    while cols < GAME_BOARD_LENGTH-1 and not merg and not merged[i][cols+1]:
                        if GAME_BOARD[i][cols+1] == GAME_BOARD[i][cols]:
                            GAME_BOARD[i][cols+1] *= 2
                            GAME_BOARD[i][cols] = None
                            merged[i][cols+1] = True
                            merg = True
                        elif GAME_BOARD[i][cols+1] is None:
                            GAME_BOARD[i][cols+1] = GAME_BOARD[i][cols]
                            GAME_BOARD[i][cols] = None
                        cols += 1

    elif direction == 2:
        for i in reversed(range(GAME_BOARD_LENGTH-1)):
            for j in range(GAME_BOARD_LENGTH):
                if GAME_BOARD[i][j] is not None:
                    rows = i
                    merg = False
                    while rows < GAME_BOARD_LENGTH-1 and not merg and not merged[rows+1][j]:
                        if GAME_BOARD[rows+1][j] == GAME_BOARD[rows][j]:
                            GAME_BOARD[rows+1][j] *= 2
                            GAME_BOARD[rows][j] = None
                            merged[rows+1][j] = True
                            merg = True
                        elif GAME_BOARD[rows+1][j] is None:
                            GAME_BOARD[rows+1][j] = GAME_BOARD[rows][j]
                            GAME_BOARD[rows][j] = None
                        rows += 1

    elif direction == 3:
        for j in range(1, GAME_BOARD_LENGTH):
            for i in range(GAME_BOARD_LENGTH):
                if GAME_BOARD[i][j] is not None:
                    cols = j
                    merg = False
                    while cols > 0 and not merg and not merged[i][cols-1]:
                        if GAME_BOARD[i][cols-1] == GAME_BOARD[i][cols]:
                            GAME_BOARD[i][cols-1] *= 2
                            GAME_BOARD[i][cols] = None
                            merged[i][cols-1] = True
                            merg = True
                        elif GAME_BOARD[i][cols-1] is None:
                            GAME_BOARD[i][cols-1] = GAME_BOARD[i][cols]
                            GAME_BOARD[i][cols] = None
                        cols -= 1

    spawn_cell()

    return get_highest_merged(merged)


def get_highest_merged(merged):
    global GAME_BOARD
    highest = 0
    for i in range(len(merged)):
        for j in range(len(merged[i])):
            value = GAME_BOARD[i][j] if GAME_BOARD[i][j] is not None else 0
            if merged[i][j] and value > highest:
                highest = value
    return highest


def get_legal_actions():
    retlist = list()
    if can_up():
        retlist.append(0)
    if can_right():
        retlist.append(1)
    if can_down():
        retlist.append(2)
    if can_left():
        retlist.append(3)
    return retlist


def get_illegal_actions():
    legal = get_legal_actions()
    return [x for x in DIRECTIONS if x not in legal]


def game_over():
    return not(can_up() or can_down() or can_left() or can_right())


def get_score():
    return sum([sum(GAME_BOARD[g]) for g in range(GAME_BOARD_LENGTH)])


def can_up():
    for r in range(GAME_BOARD_LENGTH):
        found_none = False
        for c in range(GAME_BOARD_LENGTH):
            if GAME_BOARD[c][r] is None:
                found_none = True
            elif found_none:
                return True
            elif c < GAME_BOARD_LENGTH-1 and GAME_BOARD[c+1][r] == GAME_BOARD[c][r]:
                return True
    return False


def can_down():
    for r in range(GAME_BOARD_LENGTH):
        found_none = False
        for c in reversed(range(GAME_BOARD_LENGTH)):
            if GAME_BOARD[c][r] is None:
                found_none = True
            elif found_none:
                return True
            elif c > 0 and GAME_BOARD[c-1][r] == GAME_BOARD[c][r]:
                return True
    return False


def can_right():
    for cols in range(GAME_BOARD_LENGTH):
        found_none = False
        for rows in reversed(range(GAME_BOARD_LENGTH)):
            if GAME_BOARD[cols][rows] is None:
                found_none = True
            elif found_none:
                return True
            elif rows > 0 and GAME_BOARD[cols][rows-1] == GAME_BOARD[cols][rows]:
                return True
    return False


def can_left():
    for cols in range(GAME_BOARD_LENGTH):
        found_none = False
        for rows in range(GAME_BOARD_LENGTH):
            if GAME_BOARD[cols][rows] is None:
                found_none = True
            elif found_none:
                return True
            elif rows < GAME_BOARD_LENGTH-1 and GAME_BOARD[cols][rows+1] == GAME_BOARD[cols][rows]:
                return True
    return False


def spawn_cell():
    val = 2
    if random.uniform(0, 1) > 0.9:
        val = 4
    empties = get_empties()
    if len(empties) == 0:
        return
    selection = random.choice(empties)
    GAME_BOARD[selection[0]][selection[1]] = val


def get_empties():
    empty = list()
    for i in range(len(GAME_BOARD)):
        for j in range(len(GAME_BOARD[i])):
            if GAME_BOARD[i][j] is None:
                empty.append((i, j))
    return empty


def restart():
    global GAME_BOARD
    GAME_BOARD = [[None for i in range(GAME_BOARD_LENGTH)] for j in range(GAME_BOARD_LENGTH)]
    spawn_cell()


def get_gameboard():
    return GAME_BOARD


def print_gameboard():
    print("-" * 10)
    print("\n".join([", ".join(
            [str(GAME_BOARD[z][y]) if GAME_BOARD[z][y] is not None else "0" for y in range(GAME_BOARD_LENGTH)])
         for z in range(GAME_BOARD_LENGTH)]))
    print("-" * 10)
