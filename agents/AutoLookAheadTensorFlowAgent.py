import random
from collections import deque

from Game import Game
from agents.Agent import Agent, map_state_to_inputs
import numpy as np
import tensorflow as tf

from agents.agent_tools.LookAhead import LookAhead
from rl.Episode import Episode

# Double DQN with NN switched GA lookahead


class AutoLookAheadTensorFlowAgent(Agent):

    def __init__(self, actions, features, game, exploration=0.05, alpha=0.1, gamma=0.9, experience_replays=3,
                 double_q_learning_steps=50, lookahead_prob=0.2, **kwargs):
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
        self.choices = deque()
        self.game = game
        self.lookahead_prob = lookahead_prob
        self.choice_options = [0, 1]
        self.decider = TensorFlowPerceptron(self.name + "-network1",
                                            self.features, self.actions, learning_rate=self.alpha)
        self.evaluator = TensorFlowPerceptron(self.name + "-network2",
                                              self.features, self.actions, learning_rate=self.alpha)
        self.intuition = TensorFlowPerceptron(self.name + "-intuition",
                                              self.features, self.choice_options, learning_rate=self.alpha)
        self.thinker = LookAhead(actions=actions)
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
        s = np.array(map_state_to_inputs(s)).astype(np.float)
        if len(self.action_queue) == 0:
            actions, choice = self.get_actions(s)
            self.choices.extend([choice] * len(actions))
            self.action_queue.extend(actions)
        action = self.action_queue.popleft()
        e = Episode(s, action, 0)
        e.choice = self.choices.popleft()
        self.episodes.append(e)
        return action

    def _get_ga_actions(self):
        return self.thinker.find_best(game=Game(game_board=self.game.copy_gameboard(), spawning=False))

    def get_actions(self, state):
        if random.uniform(0, 1) > self.exploration:
            actions = self.intuition.get_action(state)
            max_val = max(actions)
            action = random.choice(np.where(actions == max_val)[0])
        else:
            action = random.choice(self.choice_options)
            
        if action == 0 or (len(self.episodes) > 0 > self.episodes[-1].reward and self.episodes[-1].choice == 1):
            return self._get_e_greedy_action(state, self.exploration), action
        else:
            return self._get_ga_actions(), action

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
        self.learn_episodes(self.episodes)

    def learn_episodes(self, episodes):
        states = list()
        rewards = list()
        choice_rewards = list()
        while len(episodes) != 0:
            episode = episodes.pop(0)
            states.append(episode.state)
            ar = np.zeros(len(self.actions))
            cr = np.zeros(2)

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

        self.decider.train(states, rewards)
        self.intuition.train(states, choice_rewards)
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
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        with self.session.as_default(), self.graph.as_default():
            hidden_weights = tf.Variable(tf.constant(0., shape=[features, len(actions)]))
            self.state_ph = tf.placeholder("float", [None, features])
            self.output = tf.matmul(self.state_ph, hidden_weights)
            self.actions_ph = tf.placeholder("float", [None, len(actions)])
            loss = tf.reduce_mean(tf.square(self.output - actions))
            self.train_operation = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    def load(self):
        with self.session.as_default(), self.graph.as_default():
            saver = tf.train.Saver()
            try:
                saver.restore(self.session, "agents/models/" + self.name + ".cpkt")
            except ValueError:
                self.session.run(tf.initialize_all_variables())

    def save(self):
        with self.session.as_default(), self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.session, "agents/models/model" + self.name + ".cpkt")

    def get_action(self, state):
        with self.session.as_default(), self.graph.as_default():
            return self.session.run(self.output, feed_dict={self.state_ph: [state]})[0]

    def train(self, states, rewards):
        with self.session.as_default(), self.graph.as_default():
            self.session.run(self.train_operation, feed_dict={
                self.state_ph: states,
                self.actions_ph: rewards})
