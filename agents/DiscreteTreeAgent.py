
from agents.Agent import Agent, map_state_to_inputs, get_e_greedy_action, TreeNode
from rl.Episode import Episode


class DiscreteTreeAgent(Agent):

    def __init__(self, actions, game_size, alpha=0.1, gamma=0.9, exploration=0.05, forgetting_factor=0.9, **kwargs):
        super().__init__(actions, name="DiscreteTreeAgent", kwargs=kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.game_size = game_size
        self.exploration = exploration
        self.forgetting_factor = forgetting_factor
        self.root = TreeNode(None, self.actions)
        self.episodes = list()
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
        parent = node
        while parent.parent is not None:
            parent = parent.parent
            for i in self.actions:
                action_values[i] += parent.action_values[i]
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
            prev = get_e_greedy_action(old_values, exploration=None)

            next_node = self.episodes[0].node if len(self.episodes) != 0 else episode.node
            reward = episode.reward
            reward += self.alpha * (self.gamma * max(next_node.action_values,
                                                     key=lambda i: next_node.action_values[i]) -
                                    episode.node.action_values[episode.action])

            level = old_node.get_level()
            self._give_reward(old_node, episode.action, reward)

            new_node, values = self._get_action_values(episode.state)
            then = get_e_greedy_action(values, exploration=None)

            if reward > 0 and prev != then and level != (len(episode.state)-1) and old_node == episode.node:
                self._split_node(old_node, episode.state, level)

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
