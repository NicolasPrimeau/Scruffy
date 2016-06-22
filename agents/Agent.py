from pymongo import MongoClient


class Agent:

    def __init__(self, actions, name=None, database="AI2048", **kwargs):
        self.decision_maker = None
        self.name = name
        self.actions = actions
        self.database = database
        self.client = MongoClient()

    def load(self): pass

    def save(self): pass

    def get_action(self, state): pass

    def give_reward(self, reward): pass

    def learn(self): pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client is not None:
            self.client.close()
