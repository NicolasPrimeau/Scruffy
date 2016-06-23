import random

from pymongo import MongoClient


class Agent:

    def __init__(self, actions, name=None, database="AI2048", **kwargs):
        self.decision_maker = None
        self.name = name
        self.actions = actions
        self.database = database
        self.client = MongoClient()

    def load(self): pass

    def save(self): pass

    def get_action(self, state): pass

    def give_reward(self, reward): pass

    def learn(self): pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client is not None:
            self.client.close()


def map_state_to_inputs(state, board_size):
    state_mapping = list()
    for i in range(board_size):
        for j in range(board_size):
            key = str(i) + "_" + str(j)
            state_mapping.append(state[key])
    return state_mapping


def get_e_greedy_action(action_values, exploration):
    if exploration is not None and random.uniform(0, 1) > exploration:
        max_value = next(iter(action_values.values()))
        keys = list()
        for key, val in action_values.items():
            if val > max_value:
                keys = [key]
                max_value = val
            elif val == max_value:
                keys.append(key)
    else:
        keys = list(action_values.keys())

    return random.choice(keys) if len(keys) > 0 else "1"