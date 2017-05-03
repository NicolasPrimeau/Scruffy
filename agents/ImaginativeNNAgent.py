import random
from collections import deque

import numpy as np

from Game import Game
from agents.Agent import Agent
from agents.agent_tools.Episode import Episode
from agents.agent_tools.ExtensiveLookAhead import ExtensiveLookAhead
from agents.agent_tools.TensorFlowPerceptron import LTSMNet
from agents.agent_tools.utils import map_state_to_inputs, translate_state_to_game_board


# Double DQN with NN switched extensive lookahead


class ImaginativeNNAgent(Agent):

    def __init__(self, actions, features, exploration=0.05, alpha=0.1, gamma=0.9, experience_replays=3, **kwargs):
        super().__init__(actions, name="LookAheadTensorFlowAgent", kwargs=kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.features = features
        self.exploration = exploration
        self.episodes = list()
        self.previous = deque(maxlen=experience_replays)
        self.experience_replays = experience_replays
        self.action_queue = deque()

        self.decider = LTSMNet(self.name + "-network1", self.features, self.actions)

        self.thinker = ExtensiveLookAhead(actions=actions)
        # self.load()

    def load(self):
        self.decider.load()

    def save(self):
        self.decider.save()

    def get_action_values(self, s):
        return self.decider.predict([s])[0]

    def get_action(self, s):
        state = np.array(map_state_to_inputs(s)).astype(np.float)
        action = self._get_actions(state)[0]
        e = Episode(state, action, 0)
        self.episodes.append(e)
        return action

    def _get_lookahead_actions(self, state):
        return self.thinker.find_best(translate_state_to_game_board(state), self.get_action_values)

    def _get_actions(self, state):
        if self.exploration is not None and random.uniform(0, 1) < self.exploration:
            model = Game(game_board=translate_state_to_game_board(state), spawning=False)
            return [random.choice(model.get_legal_actions())]

        action = self._get_lookahead_actions(state)

        if action is None:
            print("LookAhead Failed")
            model = Game(game_board=translate_state_to_game_board(state), spawning=False)
            action = [random.choice(model.get_legal_actions())]

        return action

    def _get_e_greedy_action(self, state, exploration=None):
        actions = self.get_action_values(state)
        if exploration is None or (exploration is not None and random.uniform(0, 1) > exploration):
            max_val = max(actions)
            action = np.where(actions == max_val)[0]
            return [random.choice(action)]
        else:
            return [random.choice(self.actions)]

    def give_reward(self, reward):
        self.episodes[-1].reward = reward

    def _experience_replay(self):
        for episodes in self.previous:
            self.learn_episodes(list(episodes))

    def learn(self):
        self._experience_replay()
        self.previous.append(list(self.episodes))
        return self.learn_episodes(self.episodes)

    def learn_episodes(self, episodes):
        states = list()
        rewards = list()

        while len(episodes) != 0:
            episode = episodes.pop(0)
            states.append(episode.state)
            ar = np.zeros(len(self.actions))
            reward = episode.reward
            if len(episodes) != 0:
                next_episode = episodes[0]
                next_action = self._get_e_greedy_action(next_episode.state, exploration=None)
                next_actions = self.decider.predict([next_episode.state])[0]
                reward += self.gamma * next_actions[next_action]

            ar[episode.action] = reward

            rewards.append(ar.astype(float))

        self.decider.train(states, rewards)

    def clean(self):
        self.episodes[:] = []
        self.previous.clear()
