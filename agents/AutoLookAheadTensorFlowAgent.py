import copy
import random
from collections import OrderedDict
from collections import deque

import numpy as np

from Game import Game
from agents.Agent import Agent
from agents.agent_tools.Episode import Episode
from agents.agent_tools.ExtensiveLookAhead import ExtensiveLookAhead
from agents.agent_tools.LookAhead import LookAhead
from agents.agent_tools.TensorFlowPerceptron import TensorFlowPerceptron
from agents.agent_tools.utils import map_state_to_inputs


# Double DQN with NN switched GA lookahead


class AutoLookAheadTensorFlowAgent(Agent):

    def __init__(self, actions, features, exploration=0.1, alpha=0.1, gamma=0.9, experience_replays=3,
                 double_q_learning_steps=50, **kwargs):
        super().__init__(actions, name="AutoLookAheadTensorFlowAgent", kwargs=kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.features = features
        self.exploration = exploration
        self.episodes = list()
        self.previous = deque(maxlen=experience_replays)
        self.experience_replays = experience_replays
        self.dqls = double_q_learning_steps
        self.games = 0
        self.action_queue = deque()
        self.choice_queue = deque()
        self.choice_options = (0, 1)

        self.decider = TensorFlowPerceptron(self.name + "-network1",
                                            self.features, self.actions, learning_rate=self.alpha)

        self.evaluator = TensorFlowPerceptron(self.name + "-network2",
                                              self.features, self.actions, learning_rate=self.alpha)

        self.intuition = TensorFlowPerceptron(self.name + "-intuition",
                                              self.features, self.choice_options, learning_rate=self.alpha)

        self.thinker = ExtensiveLookAhead(actions=actions)
        self.load()

    def load(self):
        self.decider.load()
        self.evaluator.load()
        self.intuition.load()

    def save(self):
        self.decider.save()
        self.evaluator.save()
        self.intuition.save()

    def get_action_values(self, s):
        return self.decider.get_action(s)

    def get_action(self, s):
        state = np.array(map_state_to_inputs(s)).astype(np.float)
        if len(self.action_queue) == 0:
            actions, choice = self._get_actions(state)
            self.choice_queue.extend([choice] * len(actions))
            self.action_queue.extend(actions)
        action = self.action_queue.popleft()
        e = Episode(state, action, 0)
        e.choice = self.choice_queue.popleft()
        self.episodes.append(e)
        return action

    def _get_ga_actions(self, state):
        return self.thinker.find_best(translate_state_to_game_board(state), self.decider.get_action)

    def _get_actions(self, state):
        if random.uniform(0, 1) > self.exploration:
            actions = self.intuition.get_action(state)
            max_val = max(actions)
            choice = random.choice(np.where(actions == max_val)[0])
        else:
            choice = random.choice(self.choice_options)
                
        if choice == 0:
            action = self._get_e_greedy_action(state, exploration=self.exploration)
        elif choice == 1:
            action = self._get_ga_actions(state)
        else:
            raise ValueError("No")

        if action is None:
            print("GA Failed")
            model = Game(game_board=translate_state_to_game_board(state), spawning=False)
            action = [random.choice(model.get_legal_actions())]
            choice = 0

        return action, choice

    def _get_e_greedy_action(self, state, exploration=None):
        actions = self.get_action_values(state)
        if exploration is None or (exploration is not None and random.uniform(0, 1) > exploration):
            max_val = max(actions)
            action = np.where(actions == max_val)[0]
            return [random.choice(action)]
        else:
            return [random.choice(self.actions)]

    def give_reward(self, reward):
        if reward < 0:
            self.action_queue.clear()
            self.choice_queue.clear()
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
        choice_rewards = list()
        choices = OrderedDict.fromkeys(["title"])
        choices["title"] = "Choices"
        for option in self.choice_options:
            choices[option] = 0

        while len(episodes) != 0:
            episode = episodes.pop(0)
            states.append(episode.state)
            ar = np.zeros(len(self.actions))
            cr = np.zeros(len(self.choice_options))
            
            reward = episode.reward
            if len(episodes) != 0:
                next_episode = episodes[0]
                next_action = self._get_e_greedy_action(next_episode.state, exploration=None)
                next_actions = self.evaluator.get_action(next_episode.state)
                reward += self.gamma * next_actions[next_action]
            ar[episode.action] = reward
            cr[episode.choice] = reward
            rewards.append(ar.astype(float))
            choice_rewards.append(cr.astype(float))
            choices[episode.choice] += 1

        self.decider.train(states, rewards)
        self.intuition.train(states, choice_rewards)
        self.games += 1
        if self.games == self.dqls:
            self.games = 0
            self.decider, self.evaluator = self.evaluator, self.decider
        return choices

    def clean(self):
        self.episodes[:] = []
        self.previous.clear()


def translate_state_to_game_board(state):
    return [[2**int(item*15) if int(item*15) != 0 else None for item in row] for row in list(chunks(state, 4))]


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield copy.deepcopy(l[i:i+n])
