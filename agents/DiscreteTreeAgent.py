import random
from agents.Agent import Agent, map_state_to_inputs, get_e_greedy_action
from rl.Episode import Episode


class DiscreteTreeAgent(Agent):

    def __init__(self, actions, game_size, alpha=0.1, gamma=0.9, exploration=0.05, **kwargs):
        super().__init__(actions, name="DiscreteTreeAgent", kwargs=kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.game_size = game_size
        self.exploration = exploration
        self.root = TreeNode("Root", None, 0, self.actions)
        self.episodes = list()
        self.load()

    def load(self):
        self.root = self._recursive_load(self.root, "Root")

    def _recursive_load(self, node, key):
        record = self.client[self.database][self.name + "_tree"].find_one({"state_key": key, "level": node.level+1})
        if record is not None:
            node = TreeNode(key, node, node.level + 1, self.actions)
            node.action_values = [float(x) for x in record["action_values"]]
            for key in record["children"]:
                node.children[key] = self._recursive_load(node, key)
                node.children[key].parent = node
        return node

    def save(self):
        self._recursive_save(self.root)

    def _recursive_save(self, node):
        action_values = [str(x) for x in node.action_values]
        children = [str(key) for key in node.children.keys()]
        self.client[self.database][self.name + "_tree"].update_one({"state_key": node.state_key,
                                                                    "level": str(node.level)},
                                                                   {"$set": {"actions_values": action_values,
                                                                             "children": children}},
                                                                   upsert=True)

        for key in node.children:
            self._recursive_save(node.children[key])

    def get_action(self, state):
        state = map_state_to_inputs(state, self.game_size)
        node = self._recursive_get_leaf(self.root, 0, state)
        action = get_e_greedy_action(node.action_values, exploration=self.exploration)
        episode = Episode(state, action, 0)
        episode.node = node
        self.episodes.append(episode)
        return action

    def _recursive_get_leaf(self, node, level, state):
        if state[level] in node.children:
            return self._recursive_get_leaf(node.children[state[level]], level+1, state)
        else:
            return node

    def give_reward(self, reward):
        self.episodes[-1].reward = reward

    def learn(self):
        while len(self.episodes) > 0:
            episode = self.episodes.pop(0)
            prev_action = get_e_greedy_action(episode.node.action_values, exploration=None)

            next_node = self.episodes[0].node if len(self.episodes) != 0 else episode.node

            reward = episode.reward
            reward += self.alpha * (self.gamma * max(next_node.action_values,
                                                     key=lambda i: next_node.action_values[i]) -
                                    episode.node.action_values[episode.action])

            episode.node.action_values[episode.action] += reward

            then_action = get_e_greedy_action(episode.node.action_values, exploration=None)

            if reward >= 0 and prev_action != then_action and episode.node.level != (len(episode.state)-1):
                new_node = self._split_node(episode.node, episode.state)
                new_node.action_values[episode.action] += reward
                episode.node.action_values[episode.action] -= reward

    def _split_node(self, node, state):
        node.children[state[node.level]] = TreeNode(state[node.level+1], node, node.level+1, self.actions)
        return node.children[state[node.level]]


class TreeNode:
    def __init__(self, state_key, parent, level, actions):
        self.action_values = dict()
        for i in actions:
            self.action_values[i] = random.gauss(0, 1)
        self.state_key = state_key
        self.parent = parent
        self.level = level
        self.children = dict()
