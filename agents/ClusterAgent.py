
from agents.Agent import Agent
from agents.agent_tools.utils import get_e_greedy_action, map_state_to_inputs, Cluster
from rl.Episode import Episode


class ClusterAgent(Agent):
    def __init__(self, actions, features, alpha=0.1, gamma=0.9, exploration=0.05, **kwargs):
        super().__init__(actions, name="ClusterAgent", kwargs=kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.exploration = exploration
        self.clusters = dict()
        self.cluster_id = 0
        self.episodes = list()
        self.features = features
        self._create_cluster()

    def _create_cluster(self, state=None):
        self.clusters[self.cluster_id] = Cluster(self.actions, self.cluster_id, self.features, init_state=state)
        self.cluster_id += 1
        return self.clusters[self.cluster_id-1]

    def load(self): pass

    def save(self): pass

    def get_action(self, state):
        cluster = self._get_nearest_cluster(map_state_to_inputs(state))
        action = get_e_greedy_action(cluster.action_values, exploration=self.exploration)
        episode = Episode(state, action, None)
        episode.cluster = cluster
        self.episodes.append(episode)
        return action

    def _get_nearest_cluster(self, state):
        min_distance = self.clusters[0].get_distance(state)
        min_cluster = self.clusters[0]
        for key in self.clusters:
            distance = self.clusters[key].get_distance(state)
            if distance < min_distance:
                min_distance = distance
                min_cluster = self.clusters[key]
        return min_cluster

    def give_reward(self, reward):
        self.episodes[-1].reward = reward

    def learn(self):
        while len(self.episodes) > 0:
            episode = self.episodes.pop(0)

            next_cluster = self.episodes[0].cluster if len(self.episodes) != 0 else episode.cluster
            reward = episode.reward
            reward += self.alpha * (self.gamma * max(next_cluster.action_values,
                                                     key=lambda i: next_cluster.action_values[i]) -
                                    episode.cluster.action_values[episode.action])

            nearest = self._get_nearest_cluster(map_state_to_inputs(episode.state))
            action = get_e_greedy_action(nearest.action_values, exploration=None)

            if nearest is episode.cluster and episode.action == action and reward > 0:
                episode.cluster.action_values[episode.action] += reward
                episode.cluster.update(map_state_to_inputs(episode.state))
            elif action == episode.action and reward > 0:
                nearest.action_values[episode.action] += reward
                mapped = map_state_to_inputs(episode.state)
                nearest.update(mapped)
                nearest.remove(mapped)
            elif action != episode.action and reward > 0:
                mapped = map_state_to_inputs(episode.state)
                episode.cluster.remove(mapped)
                cluster = self._create_cluster(state=mapped)
                cluster.action_values[episode.action] += reward
            else:
                episode.cluster.action_values[episode.action] += reward

        for s in self.clusters:
            if self.clusters[s].updated < 0:
                del self.clusters[s]

        print("Num clusters" + str(len(self.clusters)))

    def clean(self):
        self.episodes[:] = []

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client is not None:
            self.client.close()
