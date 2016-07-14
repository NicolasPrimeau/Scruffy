import random
import math

MAX_VAL = 2**15
LEVELS = math.log(MAX_VAL, 2)

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



class Cluster:
    def __init__(self, actions, cid, num_features, init_state=None):
        self.updated = 0
        self.state = [0 for i in range(num_features)]
        if init_state is not None:
            self.updated += 1
            self.update(init_state)
            self.updated -= 1
        self.id = cid
        self.action_values = dict()
        for i in actions:
            self.action_values[i] = random.gauss(0, 1)

    def get_distance(self, state):
        distance = 0
        for i in range(len(self.state)):
            distance += abs(self.state[i] - state[i])
        return distance

    def update(self, state):
        self.updated += 1
        for i in range(len(self.state)):
            self.state[i] += (state[i] - self.state[i]) / float(self.updated)

    def remove(self, state):
        if self.updated == 0:
            return
        for i in range(len(self.state)):
            self.state[i] += (state[i] - self.state[i]) / float(self.updated)
        self.updated -= 1


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
    global LEVELS
    state_mapping = list()
    for i in range(int(len(state)**0.5)):
        for j in range(int(len(state)**0.5)):
            key = str(i) + "_" + str(j)
            value = math.log(state[key], 2) / LEVELS if state[key] != 0 else 0
            state_mapping.append(value)
    return state_mapping


def get_e_greedy_action(action_values, exploration=None):
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