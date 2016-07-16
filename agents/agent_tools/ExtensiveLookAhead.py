
import itertools

from Game import Game
from agents.agent_tools.utils import map_state_to_inputs


class ExtensiveLookAhead:

    def __init__(self, actions, lookahead=3, discounted=0.70):
        self.actions = actions
        self.env = None
        self.value_function = None
        self.lookahead = lookahead
        self.discounted = discounted

    def find_best(self, board, value_function):
        self.value_function = value_function
        self.env = board
        max_combo = None
        max_reward = None

        for combo in itertools.product((0, 1, 2, 3), repeat=self.lookahead):
            reward = sum(self.reward(combo))
            if max_reward is None or reward > max_reward:
                max_combo = combo
                max_reward = reward

        if max_reward is not None and max_reward > 0:
            return max_combo
        else:
            return None

    def reward(self, individual):
        game = Game(game_board=self.env, spawning=False)
        intuitive_reward = 0
        cnt = 0
        memory_reward = 0
        for action in individual:
            if action in game.get_illegal_actions():
                return -2048, memory_reward
            action_values = self.value_function(map_state_to_inputs(game.get_state()[0]))
            predicted_reward = game.do_action(action) * (self.discounted ** cnt)
            if game.game_over():
                return -2048, -2048
            intuitive_reward += predicted_reward
            memory_reward += action_values[action] * (self.discounted ** cnt)
            cnt += 1

        return intuitive_reward, memory_reward
