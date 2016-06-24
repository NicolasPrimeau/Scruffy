
from agents.Agent import Agent, map_state_to_inputs, get_e_greedy_action, TreeNode, GraphNode
from rl.Episode import Episode


class DiscreteGraphAgent(Agent):

    def __init__(self, actions, game_size, alpha=0.1, gamma=0.9, exploration=0.05, **kwargs):
        super().__init__(actions, name="DiscreteGraphAgent", kwargs=kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.game_size = game_size
        self.exploration = exploration
        self.root = GraphNode(None, self.actions, None, None)
        self.episodes = list()
        self.load()

    def load(self):
        record = self.client[self.database][self.name + "_tree"].find_one({"feature": "Root"})
        if record is not None:
            node = GraphNode(None, self.actions, None, None)
            node.action_values = dict()
            for i in self.actions:
                node.action_values[i] = float(record["actions_values"][i])

            for key in record["children"]:
                val = record["children"][key]["val"]
                nid = record["children"][key]["id"]
                level = record["children"][key]["level"]
                node.children[key] = self._recursive_load(node, val, nid, level)
            self.root = node

    def _recursive_load(self, node, f, fid, level):
        record = self.client[self.database][self.name + "_tree"].find_one({"feature": f, "feature_id": fid, "level": level})
        if record is not None:
            node = GraphNode(node, self.actions, int(f), int(fid))
            node.action_values = dict()
            for i in self.actions:
                node.action_values[i] = float(record["actions_values"][i])
            for key in record["children"]:
                val = record["children"][key]["val"]
                nid = record["children"][key]["id"]
                new_level = record["children"][key]["level"]
                node.children[key] = self._recursive_load(node, val, nid, new_level)
        return node

    def save(self): pass

    def get_action(self, state):
        state = map_state_to_inputs(state)
        leafs = self._recursive_get_leafs(self.root, state)

        max_node = max(leafs, key=lambda leaf: max(leaf.action_values))
        to_del = list(filter(lambda x: len(x.children) == 0 and x is not max_node, leafs))

        for node in to_del:
            del node.parent.children[(node.feature_id, node.feature)]
            node.parent = None

        action = get_e_greedy_action(max_node.action_values, exploration=self.exploration)
        episode = Episode(state, action, 0)
        episode.node = max_node
        episode.leafs = leafs
        self.episodes.append(episode)
        return action

    def _recursive_get_leafs(self, node, state):
        leafs = list()
        for child in node.get_next(state):
            leafs.extend(self._recursive_get_leafs(child, state))
        if len(leafs) == 0:
            leafs.append(node)
        return leafs

    def give_reward(self, reward):
        self.episodes[-1].reward = reward

    def learn(self):
        split_nodes = list()
        while len(self.episodes) > 0:
            episode = self.episodes.pop(0)
            if episode.node in split_nodes:
                continue

            prev_action = get_e_greedy_action(episode.node.action_values, exploration=None)
            next_node = self.episodes[0].node if len(self.episodes) != 0 else episode.node
            reward = episode.reward
            reward += self.alpha * (self.gamma * max(next_node.action_values,
                                                     key=lambda i: next_node.action_values[i]) -
                                    episode.node.action_values[episode.action])
            episode.node.action_values[episode.action] += reward

            then_action = get_e_greedy_action(episode.node.action_values, exploration=None)

            #level = episode.node.get_level()
            if prev_action != then_action: # and level != (len(episode.state)-1):
                self._split_node(episode.node, episode.state)
                episode.node.action_values[episode.action] -= reward
                split_nodes.append(episode.node)

    def _split_node(self, node, state):
        for i in range(len(state)):
            new_node = GraphNode(node, self.actions, state[i], i)
            new_node.action_values = node.action_values.copy()
            node.children[(i, state[i])] = new_node

    def print_tree(self):
        self._recursive_print_tree(self.root, "")

    def _recursive_print_tree(self, node, padding):
        print(padding + str(node.feature) + ", " + str(node.feature_id))
        for key in node.children:
            self._recursive_print_tree(node.children[key], padding + "\t")
