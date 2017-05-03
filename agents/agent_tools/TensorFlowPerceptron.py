
from keras.layers import Dense, LSTM
from keras.models import Sequential
import numpy as np
import os
from keras.models import load_model


class BasicNet:

    def __init__(self, name, features, outputs, load=False):
        self.name = name
        self.features = features
        if not load:
            self.model = Sequential()
            self.model.add(Dense(self.features, activation="sigmoid", input_shape=(features,)))
            self.model.add(Dense(self.features, activation="sigmoid"))
            self.model.add(Dense(len(outputs), activation="sigmoid"))
            self.model.compile(loss='mean_squared_error', optimizer='adam')
        else:
            self.load()

    def forget(self):
        self.model.reset_states()

    def train(self, train, rewards, verbose=0, epochs=100):
        train = np.array(train)
        rewards = np.array(rewards)
        self.model.fit(train, rewards, verbose=verbose, epochs=epochs)

    def predict(self, states):
        return self.model.predict(np.array(states))

    def save(self):
        self.model.save("data/models/" + self.name + ".h5")
        print("Sucessfully Saved Model")
        return True

    def load(self):
        if os.path.exists("data/models/" + self.name + ".h5"):
            self.model = load_model("data/models/" + self.name + ".h5")
            self.model.reset_states()
            print("Sucessfully Loaded Model")
        return True



class LTSMNet:

    def __init__(self, name, features, outputs, load=False):
        self.name = name
        self.features = features
        if not load:
            self.model = Sequential()
            self.model.add(LSTM(self.features, input_shape=(1, features)))
            # self.model.add(LSTM(self.features, input_shape=(1, features)))
            self.model.add(Dense(len(outputs), activation='relu'))
            # self.model.add(Dense())
            self.model.compile(loss='mean_squared_error', optimizer='adam')
        else:
            self.load()

    def forget(self):
        self.model.reset_states()

    def train(self, train, rewards, verbose=0, epochs=100):
        train = np.array([[x] for x in train])
        rewards = np.array(rewards)
        self.model.fit(train, rewards, verbose=verbose, epochs=epochs)

    def predict(self, states):
        return self.model.predict(np.array([states]))

    def save(self):
        self.model.save("data/models/" + self.name + ".h5")
        print("Sucessfully Saved Model")
        return True

    def load(self):
        if os.path.exists("data/models/" + self.name + ".h5"):
            self.model = load_model("data/models/" + self.name + ".h5")
            self.model.reset_states()
            print("Sucessfully Loaded Model")
        return True

