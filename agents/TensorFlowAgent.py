import random

from agents.Agent import Agent, map_state_to_inputs
import numpy as np
import tensorflow as tf

from rl.Episode import Episode


class TensorFlowAgent(Agent):

    def __init__(self, actions, features, exploration=0.05, alpha=0.1, gamma=0.9, **kwargs):
        super().__init__(actions, name="TensorFlowAgent", kwargs=kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.features = features
        self.exploration = exploration
        self.episodes = list()

        session = tf.Session()
        hidden_weights = tf.Variable(tf.constant(0., shape=[self.features, len(self.actions)]))
        self.state_ph = tf.placeholder("float", [None, self.features])
        self.output = tf.matmul(self.state_ph, hidden_weights)
        self.actions_ph = tf.placeholder("float", [None, len(self.actions)])
        loss = tf.reduce_mean(tf.square(self.output - actions))
        self.train_operation = tf.train.AdamOptimizer(learning_rate=self.alpha).minimize(loss)
        session.run(tf.initialize_all_variables())
        self.session = session

    def load(self):
        pass

    def save(self):
        pass

    def get_action(self, s):
        s = np.array(map_state_to_inputs(s)).astype(np.float)
        actions = self.session.run(self.output,
                                   feed_dict={self.state_ph: [s]})[0]
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

    def learn(self):
        states = list()
        rewards = list()
        while len(self.episodes) != 0:
            episode = self.episodes.pop(0)
            states.append(episode.state)
            ar = np.zeros(4)

            next_state = self.episodes[0].state if len(self.episodes) != 0 else episode.state
            next_actions = self.session.run(self.output, feed_dict={self.state_ph: [next_state]})[0]

            reward = episode.reward
            reward += self.alpha * (self.gamma * max(next_actions) - episode.actions[episode.action])
            ar[episode.action] = reward

            rewards.append(ar.astype(float))

        self.session.run(self.train_operation, feed_dict={
                self.state_ph: states,
                self.actions_ph: rewards})
