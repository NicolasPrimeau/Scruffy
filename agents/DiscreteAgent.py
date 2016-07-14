import random

from agents.Agent import Agent
from agents.agent_tools.utils import get_e_greedy_action
from rl.Episode import Episode


class DiscreteAgent(Agent):

    def __init__(self, actions, alpha=0.1, gamma=0.9, exploration=0.05, elligibility_trace=False,
                 forgetting_factor=0.9, **kwargs):
        super().__init__(actions, name="DiscreteAgent", kwargs=kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.exploration = exploration
        self.forgetting_factor = forgetting_factor
        self.episodes = list()
        self.eligibility_trace = elligibility_trace

    def load(self):
        pass

    def save(self):
        pass

    def get_action(self, state):
        if state is None:
            return random.choice(self.actions)

        database = self.client[self.database]
        if database is None:
            return random.choice(self.actions)
        states = database[self.name + "_states"]
        if states is None:
            record = create_new_entry(state, states, self.actions)
        else:
            record = states.find_one({"state": state})

        if record is None:
            record = create_new_entry(state, states, self.actions)

        action = get_e_greedy_action(record["actions"], self.exploration)
        self.episodes.append(Episode(state, action, 0))
        return action

    def give_reward(self, reward):
        self.episodes[-1].reward = reward

    def learn(self):
        if self.eligibility_trace:
            eligibles = list()

        while len(self.episodes) > 0:
            episode = self.episodes.pop(0)
            database = self.client[self.database]
            if database is None:
                return

            states = database[self.name + "_states"]
            if states is None:
                return

            record = states.find_one({"state": episode.state})
            if record is None:
                return

            next_state = self.episodes[0].state if len(self.episodes) != 0 else episode.state

            next_record = states.find_one({"state": next_state})
            if next_record is None:
                next_record = create_new_entry(next_state, states, self.actions)

            reward = episode.reward
            reward += self.alpha * (self.gamma * next_record["actions"][str(max(next_record["actions"],
                                                    key=next_record["actions"].get))] - record["actions"][episode.action])
            record["actions"][episode.action] += reward
            states.update({'_id': record['_id']}, record)

            if self.eligibility_trace:
                for idx in range(len(eligibles)):
                    discounted = (self.forgetting_factor**idx) * reward
                    if discounted < 0.001:
                        break
                    eligibles[-idx][0]["actions"][eligibles[-idx][1]] += discounted
                    states.update({'_id': eligibles[-idx][0]['_id']}, eligibles[-idx][0])
                eligibles.append((record, episode.action))

    def clean(self):
        self.episodes[:] = []


def create_new_entry(state, collection, actions):
    new_entry = dict()
    new_entry["state"] = state
    new_entry["actions"] = dict()
    for i in actions:
        new_entry["actions"][str(i)] = random.gauss(0, 1)
    collection.insert_one(new_entry)
    return new_entry
