
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
from keras.models import load_model


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
        # self.train(np.array([[[0] * 16]]), np.array([[0,0,0,0]]))

    def forget(self):
        self.model.reset_states()

    def train(self, train, rewards, verbose=0, epochs=1000):
        train = np.array([[x] for x in train])
        rewards = np.array(rewards)
        self.model.fit(train, rewards, verbose=verbose, epochs=epochs)

    def predict(self, states):
        return self.model.predict(np.array([states]))

    def save(self):
        self.model.save("data/models/" + self.name, "net.h5")
        print("Sucessfully Saved Model")
        return True

    def load(self):
        self.model = load_model("data/models/" + self.name, "net.h5")
        self.model.reset_states()
        print("Sucessfully Loaded Model")
        return True

