
import random

from agents.Agent import Agent, get_e_greedy_action, map_state_to_inputs
from rl.Episode import Episode


class DiscreteNeighbourAgent(Agent):
    def __init__(self, actions, alpha=0.1, gamma=0.9, exploration=0.05, **kwargs):
        super().__init__(actions, name="DiscreteNeighbourAgent", kwargs=kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.exploration = exploration
        self.clusters = list()
        self.episodes = list()

    def _create_cluster(self, state):
        for cluster in self.clusters:
            if compute_distance(cluster, state) == 0:
                return
        self.clusters.append(state)

    def load(self): pass

    def save(self): pass

    def get_action(self, state):
        state = map_state_to_inputs(state)
        neighbours = self._get_nearest_neighbours(state)
        database = self.client[self.database]
        states = database[self.name + "_states"]
        if len(neighbours) == 0:
            self._create_cluster(state)
            record = create_new_entry(state, states, self.actions)
            action = get_e_greedy_action(record["actions"], self.exploration)
            episode = Episode(state, action, 0)
            episode.action_value = record["actions"][action]
            episode.neighbours = [(0, state)]
            self.episodes.append(episode)
            return action

        action_values = [0 for i in range(len(self.actions))]
        for neighbour in neighbours:
            record = states.find_one({"state": neighbour[1]})
            for i in self.actions:
                action_values[i] += record["actions"][str(i)]

        action, value = self.get_max_action(action_values, exploration=self.exploration)
        episode = Episode(state, action, 0)
        episode.action_value = value
        episode.neighbours = neighbours
        self.episodes.append(episode)
        return action

    def get_max_action(self, action_values, exploration=None):
        if exploration is not None and random.uniform(0, 1) < exploration:
            action = random.choice(self.actions)
        else:
            action = action_values.index(max(action_values))
        return action, action_values[action]

    def _get_nearest_neighbours(self, state, k=5):
        nearest = list()
        for neighbour in self.clusters:
            distance = compute_distance(state, neighbour)
            if len(nearest) < k or distance < nearest[-1][0]:
                nearest.append((distance, neighbour))
                nearest.sort(key=lambda x: x[0])
                if len(nearest) > k:
                    nearest = nearest[:k]

        return nearest

    def give_reward(self, reward):
        self.episodes[-1].reward = reward

    def learn(self):
        while len(self.episodes) > 0:
            episode = self.episodes.pop(0)

            next_record = self.episodes[0] if len(self.episodes) != 0 else episode

            reward = episode.reward
            reward += self.alpha * (self.gamma * next_record.action_value -episode.action_value)
            database = self.client[self.database]
            states = database[self.name + "_states"]
            if reward < 0:
                self._create_cluster(episode.state)
                record = create_new_entry(episode.state, states, self.actions)

            for neighbour in episode.neighbours:
                neighbour = neighbour[1]
                record = states.find_one({"state": neighbour})
                record["actions"][str(episode.action)] += reward
                states.update({'_id': record['_id']}, record)


def create_new_entry(state, collection, actions):
    new_entry = dict()
    new_entry["state"] = state
    new_entry["actions"] = dict()
    for i in actions:
        new_entry["actions"][str(i)] = random.gauss(0, 1)
    collection.insert_one(new_entry)
    return new_entry


def compute_distance(s1, s2):
    distance = 0
    for idx in range(len(s1)):
        distance += abs(s1[idx] - s2[idx])
    return distance
