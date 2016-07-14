
from agents.Agent import Agent
from agents.agent_tools.utils import TreeNode, map_state_to_inputs, get_e_greedy_action
from rl.Episode import Episode


class DiscreteTreeAgent(Agent):

    def __init__(self, actions, game_size, alpha=0.1, gamma=0.9, exploration=0.05, forgetting_factor=0.9,
                 pruning=10000, **kwargs):
        super().__init__(actions, name="DiscreteTreeAgent", kwargs=kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.game_size = game_size
        self.exploration = exploration
        self.forgetting_factor = forgetting_factor
        self.root = TreeNode(None, self.actions)
        self.episodes = list()
        self.pruning = pruning
        self.pruning_count = 0
        self.load()

    def load(self):
        self.root = self._recursive_load(self.root, self.root.get_feature(), 0)
        self.root.parent = None

    def _recursive_load(self, node, state_key, level):
        record = self.client[self.database][self.name + "_tree"].find_one({"state_key": state_key, "level": str(level)})
        if record is not None:
            node = TreeNode(node, self.actions)
            node.action_values = dict()
            for i in self.actions:
                node.action_values[i] = float(record["actions_values"][i])
            for key in record["children"]:
                node.children[key] = self._recursive_load(node, key, level+1)
                node.children[key].parent = node
        return node

    def save(self):
        self._recursive_save(self.root, self.root.get_feature(), 0)

    def _recursive_save(self, node, state_key, level):
        action_values = [str(x) for x in node.action_values]
        children = [str(key) for key in node.children.keys()]
        self.client[self.database][self.name + "_tree"].update_one({"state_key": state_key,
                                                                    "level": str(level)},
                                                                   {"$set": {"actions_values": action_values,
                                                                             "children": children}},
                                                                   upsert=True)

        for key in node.children:
            self._recursive_save(node.children[key], key, level+1)

    def get_action(self, state):
        state = map_state_to_inputs(state)
        node, action_values = self._get_action_values(state)
        action = get_e_greedy_action(action_values, exploration=self.exploration)
        episode = Episode(state, action, 0)
        episode.node = node
        self.episodes.append(episode)
        return action

    def _get_action_values(self, state):
        node = self._recursive_get_leaf(self.root, state, 0)
        action_values = dict(node.action_values)
        #parent = node
        #while parent.parent is not None:
        #    parent = parent.parent
        #    for i in self.actions:
        #        action_values[i] += parent.action_values[i]
        return node, action_values

    def _recursive_get_leaf(self, node, state, level):
        if state[level] in node.children:
            return self._recursive_get_leaf(node.children[state[level]], state, level+1)
        else:
            return node

    def give_reward(self, reward):
        self.episodes[-1].reward = reward

    def learn(self):

        while len(self.episodes) > 0:
            episode = self.episodes.pop(0)

            old_node, old_values = self._get_action_values(episode.state)
            next_state = self.episodes[0].state if len(self.episodes) != 0 else episode.state
            next_node, next_values = self._get_action_values(next_state)
            reward = episode.reward
            reward += self.alpha * (self.gamma * max(next_values, key=lambda i: next_values[i]) -
                                    old_values[episode.action])
            level = old_node.get_level()

            if reward > 0 and episode.node is old_node and level != (len(episode.state)-1):
                new = self._split_node(old_node, episode.state, level)
                new.action_values[episode.action] += reward
            else:
                self._give_reward(old_node, episode.action, reward)

        self.pruning_count += 1
        if self.pruning is not None and self.pruning_count == self.pruning:
            self._prune(self.root)
            self.pruning_count = 0

    def _prune(self, node, parent_values=None):
        action_values = dict(node.action_values)
        if parent_values is not None:
            for key in parent_values:
                action_values[key] += parent_values[key]

        to_del = list()
        for key in node.children:
            if self._prune(node.children[key], action_values):
                to_del.append(key)

        if parent_values is None:
            return

        action = max(action_values, key=lambda i: action_values[i])
        parent_action = max(parent_values, key=lambda i: parent_values[i])

        for key in to_del:
            node.children[key].parent = None
            del node.children[key]

        return node.parent is not None and len(node.children) == 0 and action == parent_action

    def _give_reward(self, node, action, reward):
        node.action_values[action] += reward
        cnt = 1
        while node.parent is not None:
            node = node.parent
            node.action_values[action] += reward * (self.forgetting_factor**cnt)
            cnt += 1

    def _split_node(self, node, state, level):
        node.children[state[level]] = TreeNode(node, self.actions)
        node.children[state[level]].action_values = node.action_values.copy()
        return node.children[state[level]]

    def clean(self):
        self.episodes[:] = []
