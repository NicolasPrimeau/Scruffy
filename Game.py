import copy
import random

GAME_BOARD_LENGTH = 4

PROB_2 = 0.9

# Up right down Left
DIRECTIONS = [0, 1, 2, 3]


class Game:
    def __init__(self, game_board=None, spawning=True):
        if game_board is None:
            self.game_board = [[None for i in range(GAME_BOARD_LENGTH)] for j in range(GAME_BOARD_LENGTH)]
        else:
            self.game_board = copy_gameboard(game_board)
        self.spawning = spawning

    def do_action(self, direction):
        direction = int(direction)
        if direction not in DIRECTIONS:
            return 0

        merged = [[False for j in range(GAME_BOARD_LENGTH)] for i in range(GAME_BOARD_LENGTH)]

        if direction == 0:
            for i in range(1, GAME_BOARD_LENGTH):
                for j in range(GAME_BOARD_LENGTH):
                    if self.game_board[i][j] is not None:
                        rows = i
                        merg = False
                        while rows > 0 and not merg and not merged[rows-1][j]:
                            if self.game_board[rows-1][j] == self.game_board[rows][j]:
                                self.game_board[rows - 1][j] *= 2
                                self.game_board[rows][j] = None
                                merged[rows-1][j] = True
                                merg = True
                            elif self.game_board[rows-1][j] is None:
                                self.game_board[rows - 1][j] = self.game_board[rows][j]
                                self.game_board[rows][j] = None
                            rows -= 1

        elif direction == 1:
            for j in reversed(range(0, GAME_BOARD_LENGTH-1)):
                for i in range(GAME_BOARD_LENGTH):
                    if self.game_board[i][j] is not None:
                        cols = j
                        merg = False
                        while cols < GAME_BOARD_LENGTH-1 and not merg and not merged[i][cols+1]:
                            if self.game_board[i][cols+1] == self.game_board[i][cols]:
                                self.game_board[i][cols + 1] *= 2
                                self.game_board[i][cols] = None
                                merged[i][cols+1] = True
                                merg = True
                            elif self.game_board[i][cols+1] is None:
                                self.game_board[i][cols + 1] = self.game_board[i][cols]
                                self.game_board[i][cols] = None
                            cols += 1

        elif direction == 2:
            for i in reversed(range(GAME_BOARD_LENGTH-1)):
                for j in range(GAME_BOARD_LENGTH):
                    if self.game_board[i][j] is not None:
                        rows = i
                        merg = False
                        while rows < GAME_BOARD_LENGTH-1 and not merg and not merged[rows+1][j]:
                            if self.game_board[rows+1][j] == self.game_board[rows][j]:
                                self.game_board[rows + 1][j] *= 2
                                self.game_board[rows][j] = None
                                merged[rows+1][j] = True
                                merg = True
                            elif self.game_board[rows+1][j] is None:
                                self.game_board[rows + 1][j] = self.game_board[rows][j]
                                self.game_board[rows][j] = None
                            rows += 1

        elif direction == 3:
            for j in range(1, GAME_BOARD_LENGTH):
                for i in range(GAME_BOARD_LENGTH):
                    if self.game_board[i][j] is not None:
                        cols = j
                        merg = False
                        while cols > 0 and not merg and not merged[i][cols-1]:
                            if self.game_board[i][cols-1] == self.game_board[i][cols]:
                                self.game_board[i][cols - 1] *= 2
                                self.game_board[i][cols] = None
                                merged[i][cols-1] = True
                                merg = True
                            elif self.game_board[i][cols-1] is None:
                                self.game_board[i][cols - 1] = self.game_board[i][cols]
                                self.game_board[i][cols] = None
                            cols -= 1

        if self.spawning:
            self.spawn_cell()

        return self.get_highest_merged(merged)

    def get_highest_merged(self, merged):
        highest = 0
        for i in range(len(merged)):
            for j in range(len(merged[i])):
                value = self.game_board[i][j] if self.game_board[i][j] is not None else 0
                if merged[i][j] and value > highest:
                    highest = value
        return highest

    def get_legal_actions(self):
        retlist = list()
        if self.can_up():
            retlist.append(0)
        if self.can_right():
            retlist.append(1)
        if self.can_down():
            retlist.append(2)
        if self.can_left():
            retlist.append(3)
        return retlist

    def get_illegal_actions(self):
        legal = self.get_legal_actions()
        return [x for x in DIRECTIONS if x not in legal]

    def game_over(self):
        return not(self.can_up() or self.can_down() or self.can_left() or self.can_right())

    def get_score(self):
        return sum([sum(self.game_board[g]) for g in range(GAME_BOARD_LENGTH)])

    def can_up(self):
        for r in range(GAME_BOARD_LENGTH):
            found_none = False
            for c in range(GAME_BOARD_LENGTH):
                if self.game_board[c][r] is None:
                    found_none = True
                elif found_none:
                    return True
                elif c < GAME_BOARD_LENGTH-1 and self.game_board[c+1][r] == self.game_board[c][r]:
                    return True
        return False

    def can_down(self):
        for r in range(GAME_BOARD_LENGTH):
            found_none = False
            for c in reversed(range(GAME_BOARD_LENGTH)):
                if self.game_board[c][r] is None:
                    found_none = True
                elif found_none:
                    return True
                elif c > 0 and self.game_board[c-1][r] == self.game_board[c][r]:
                    return True
        return False

    def can_right(self):
        for cols in range(GAME_BOARD_LENGTH):
            found_none = False
            for rows in reversed(range(GAME_BOARD_LENGTH)):
                if self.game_board[cols][rows] is None:
                    found_none = True
                elif found_none:
                    return True
                elif rows > 0 and self.game_board[cols][rows-1] == self.game_board[cols][rows]:
                    return True
        return False

    def can_left(self):
        for cols in range(GAME_BOARD_LENGTH):
            found_none = False
            for rows in range(GAME_BOARD_LENGTH):
                if self.game_board[cols][rows] is None:
                    found_none = True
                elif found_none:
                    return True
                elif rows < GAME_BOARD_LENGTH-1 and self.game_board[cols][rows+1] == self.game_board[cols][rows]:
                    return True
        return False

    def spawn_cell(self):
        val = 2
        if random.uniform(0, 1) > 0.9:
            val = 4
        empties = self.get_empties()
        if len(empties) == 0:
            return
        selection = random.choice(empties)
        self.game_board[selection[0]][selection[1]] = val

    def get_empties(self):
        empty = list()
        for i in range(len(self.game_board)):
            for j in range(len(self.game_board[i])):
                if self.game_board[i][j] is None:
                    empty.append((i, j))
        return empty

    def restart(self):
        self.game_board = [[None for i in range(GAME_BOARD_LENGTH)] for j in range(GAME_BOARD_LENGTH)]
        self.spawn_cell()

    def get_gameboard(self):
        return tuple([tuple(row) for row in self.game_board])

    def copy_gameboard(self):
        return [copy.deepcopy(self.game_board[i]) for i in range(len(self.game_board))]

    def print_gameboard(self):
        print("-" * 10)
        print("\n".join([", ".join(
                [str(self.game_board[z][y]) if self.game_board[z][y] is not None else "0" for y in
                 range(GAME_BOARD_LENGTH)])
             for z in range(GAME_BOARD_LENGTH)]))
        print("-" * 10)

    def get_state(self):
        state = dict()
        score = 0
        gameboard = self.copy_gameboard()
        for i in range(len(gameboard)):
            for j in range(len(gameboard[i])):
                name = str(i) + "_" + str(j)
                state[name] = gameboard[i][j] if gameboard[i][j] is not None else 0
                score += state[name]
        return state, score


def copy_gameboard(board):
    return [copy.deepcopy(board[i]) for i in range(len(board))]
