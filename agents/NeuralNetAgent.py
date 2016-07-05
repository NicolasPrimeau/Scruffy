import pickle
from bson import Binary
from pybrain import SigmoidLayer, LinearLayer
from pybrain.rl.agents import LearningAgent

from pybrain.rl.learners import ActionValueNetwork, NFQ
from pybrain.tools.shortcuts import buildNetwork

from agents.Agent import Agent, map_state_to_inputs


class NeuralNetAgent(Agent):

    def __init__(self, actions, game_size, alpha=0.1, gamma=0.9, **kwargs):
        super().__init__(actions, name="NeuralNetAgent", kwargs=kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.features = game_size**2
        self.agent = None
        self.load()

    def load(self):
        record = self.client[self.database].models.find_one({"model": self.name})
        if record is None:
            print("No Network found in DB, creating new one")
            controller = ActionValueNetwork(self.features, len(self.actions))

            controller.network = buildNetwork(self.features + len(self.actions),
                                              self.features + len(self.actions),
                                              self.features + len(self.actions),
                                              1)

            controller.network.is_loaded_correctly = True
        else:
            controller = ActionValueNetwork(self.features, len(self.actions))
            controller.network = pickle.loads(record["data"])
            controller.network.sorted = False
            controller.network.sortModules()
            try:
                if not controller.network.is_loaded_correctly:
                    print("Pre-existing neural network data corrupted, creating new")
                    controller = ActionValueNetwork(self.features, len(self.actions))
                    controller.network.is_loaded_correctly = True
            except AttributeError:
                print("Neural Network did not load correctly, creating new")
                controller = ActionValueNetwork(self.features, len(self.actions))
                controller.network.is_loaded_correctly = True
            controller.network.is_loaded_correctly = True
        self.agent = LearningAgent(controller, NFQ(alpha=self.alpha, gamma=self.gamma))
        self.agent.newEpisode()

    def save(self):
        self.client[self.database]["models"].update_one({"model": self.name},
                                             {"$set": {"data": Binary(pickle.dumps(self.agent.module.network))}},
                                             upsert=True)

    def get_action(self, state):
        self.agent.integrateObservation(map_state_to_inputs(state))
        return int(self.agent.getAction()[0])

    def give_reward(self, reward):
        self.agent.giveReward(reward/2048)

    def learn(self):
        self.agent.learn()
        self.agent.newEpisode()
        self.save()
