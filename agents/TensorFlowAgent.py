import random
from collections import deque

from Game import Game
from agents.Agent import Agent, map_state_to_inputs
import numpy as np
import tensorflow as tf

from rl.Episode import Episode

# Double DQN


class TensorFlowAgent(Agent):

    def __init__(self, actions, features, exploration=0.05, alpha=0.1, gamma=0.9, experience_replays=4,
                 double_q_learning_steps=100, **kwargs):
        super().__init__(actions, name="TensorFlowAgent", kwargs=kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.features = features
        self.exploration = exploration
        self.episodes = list()
        self.previous = deque(maxlen=experience_replays)
        self.experience_replays = experience_replays
        self.dqls = double_q_learning_steps
        self.games = 0

        self.decider = TensorFlowPerceptron("network1", self.features, self.actions, learning_rate=self.alpha)
        self.evaluator = TensorFlowPerceptron("network2", self.features, self.actions, learning_rate=self.alpha)
        self.load()

    def load(self):
        self.decider.load()
        self.evaluator.load()

    def save(self):
        self.decider.save()
        self.evaluator.load()

    def get_action(self, s):
        s = np.array(map_state_to_inputs(s)).astype(np.float)
        actions = self.decider.get_action(s)
        action = self._get_e_greedy_action(actions, self.exploration)
        e = Episode(s, action, 0)
        e.actions = actions
        self.episodes.append(e)
        return action

    def _get_e_greedy_action(self, actions, exploration):
        if exploration is None or (exploration is not None and random.uniform(0, 1) > exploration):
            max_val = max(actions)
            action = np.where(actions == max_val)[0]
            return random.choice(action)
        else:
            return random.choice(self.actions)

    def give_reward(self, reward):
        self.episodes[-1].reward = reward

    def _experience_replay(self):
        for episodes in self.previous:
            self.learn_episodes(list(episodes))

    def learn(self):
        self._experience_replay()
        self.previous.append(list(self.episodes))
        self.learn_episodes(self.episodes)

    def learn_episodes(self, episodes):
        states = list()
        rewards = list()
        while len(episodes) != 0:
            episode = episodes.pop(0)
            states.append(episode.state)
            ar = np.zeros(4)

            reward = episode.reward
            if len(episodes) != 0:
                next_episode = episodes[0]
                next_action = self._get_e_greedy_action(next_episode.actions, exploration=None)
                next_actions = self.evaluator.get_action(next_episode.state)
                reward += self.gamma * next_actions[next_action]
            ar[episode.action] = reward
            rewards.append(ar.astype(float))

        self.decider.train(states, rewards)
        self.games += 1
        if self.games == self.dqls:
            self.games = 0
            self.decider, self.evaluator = self.evaluator, self.decider

    def clean(self):
        self.episodes[:] = []
        self.previous.clear()


class TensorFlowPerceptron:

    def __init__(self, name, features, actions, learning_rate=0.1):
        self.name = name
        self.session = tf.Session()
        hidden_weights = tf.Variable(tf.constant(0., shape=[features, len(actions)]))
        self.state_ph = tf.placeholder("float", [None, features])
        self.output = tf.matmul(self.state_ph, hidden_weights)
        self.actions_ph = tf.placeholder("float", [None, len(actions)])
        loss = tf.reduce_mean(tf.square(self.output - actions))
        self.train_operation = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    def load(self):
        saver = tf.train.Saver()
        try:
            saver.restore(self.session, "agents/models/" + self.name + ".cpkt")
        except ValueError:
            self.session.run(tf.initialize_all_variables())

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.session, "agents/models/model" + self.name + ".cpkt")

    def get_action(self, state):
        return self.session.run(self.output, feed_dict={self.state_ph: [state]})[0]

    def train(self, states, rewards):
        self.session.run(self.train_operation, feed_dict={
            self.state_ph: states,
            self.actions_ph: rewards})
