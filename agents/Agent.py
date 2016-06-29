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


class TreeNode:
    def __init__(self, parent, actions):
        self.action_values = dict()
        for i in actions:
            self.action_values[i] = random.gauss(0, 1)
        self.parent = parent
        self.children = dict()

    def get_level(self):
        node = self
        level = 0
        while node.parent is not None:
            node = node.parent
            level += 1
        return level

    def get_feature(self):
        if self.parent is None:
            return "Root"
        else:
            for key in self.parent.children:
                if self.parent.children[key] is self:
                    return key


class GraphNode(TreeNode):

    def __init__(self, parent, actions, feature, feature_id):
        super().__init__(parent, actions)
        self.feature = feature
        self.feature_id = feature_id

    def get_feature(self):
        return self.feature

    def get_feature_id(self):
        return self.feature_id

    def get_next(self, state):
        next_nodes = list()
        if self.feature is not None and self.feature != state[self.feature_id] or len(self.children) != 0:
            for i in range(len(state)):
                key = (i, state[i])
                if key in self.children:
                    next_nodes.append(self.children[key])
        return next_nodes


def map_state_to_inputs(state):
    state_mapping = list()
    for i in range(int(len(state)**0.5)):
        for j in range(int(len(state)**0.5)):
            key = str(i) + "_" + str(j)
            state_mapping.append(state[key])
    return state_mapping


def get_e_greedy_action(action_values, exploration):
    if exploration is None or (exploration is not None and random.uniform(0, 1) > exploration):
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