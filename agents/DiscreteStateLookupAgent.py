import random

from agents.Agent import Agent
from agents.agent_tools.Episode import Episode
from agents.agent_tools.utils import map_state_to_inputs


class DiscreteStateLookupAgent(Agent):
    def __init__(self, actions, features, alpha=0.1, gamma=0.9, exploration=0.05, **kwargs):
        super().__init__(actions, name="DiscreteStateLookupAgent", kwargs=kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.num_features = features
        self.exploration = exploration
        self.feature_table = [dict() for j in range(features)]
        self.episodes = list()
        self.load()

    def load(self): pass

    def save(self): pass

    def get_action(self, state):
        state = map_state_to_inputs(state)
        self._setup_feature_table(state)
        action, _ = self.get_max_action(state, exploration=self.exploration)

        episode = Episode(state, action, 0)
        self.episodes.append(episode)
        return action

    def _setup_feature_table(self, state):
        for i in range(len(self.feature_table)):
            if state[i] not in self.feature_table[i]:
                self.feature_table[i][state[i]] = [random.gauss(0, 1) for i in range(len(self.actions))]

    def give_reward(self, reward):
        self.episodes[-1].reward = reward

    def learn(self):
        while len(self.episodes) > 0:
            episode = self.episodes.pop(0)

            action_value = 0
            for idx in range(len(self.feature_table)):
                action_value += self.feature_table[idx][episode.state[idx]][episode.action]

            next_record = self.episodes[0] if len(self.episodes) != 0 else episode
            next_action, next_value = self.get_max_action(next_record.state)

            reward = episode.reward
            reward += self.alpha * (self.gamma * next_value - action_value)

            for idx in range(len(self.feature_table)):
                self.feature_table[idx][episode.state[idx]][episode.action] += reward

    def lookup_state(self, state):
        action_values = [0 for i in range(len(self.actions))]
        for idx in range(len(self.feature_table)):
            for action in self.actions:
                action_values[action] += self.feature_table[idx][state[idx]][action]
        return action_values

    def get_max_action(self, state, exploration=None):
        action_values = self.lookup_state(state)
        if exploration is not None and random.uniform(0, 1) < exploration:
            action = random.choice(self.actions)
        else:
            action = action_values.index(max(action_values))
        return action, action_values[action]

    def clean(self):
        self.episodes[:] = []
