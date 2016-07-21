import tensorflow as tf


class TensorFlowPerceptron:

    def __init__(self, name, features, actions, learning_rate=0.1):
        self.name = name
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        self.loaded = False

        with self.session.as_default(), self.graph.as_default():
            self.state_ph = tf.placeholder("float", [None, features])
            self.actions_ph = tf.placeholder("float", [None, len(actions)])
            w_h = self.init_weights([features, features])
            # w_h_2 = self.init_weights([features, features])
            w_o = self.init_weights([features, len(actions)])
            self.network = self.model(self.state_ph, [w_h], w_o)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.network, self.actions_ph))
            self.train_operation = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    @staticmethod
    def init_weights(shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

    @staticmethod
    def model(x, w_h, w_o):
        h = tf.nn.tanh(tf.matmul(x, w_h[0]))
        if len(w_h) > 1:
            for hidden in w_h[1:]:
                h = tf.nn.tanh(tf.matmul(h, hidden))
        return tf.matmul(h, w_o)

    def load(self):
        with self.session.as_default(), self.graph.as_default():
            saver = tf.train.Saver()
            try:
                saver.restore(self.session, "agents/models/" + self.name + ".cpkt")
                print(self.name + " Loaded Successfully")
            except ValueError:
                print(self.name + " No Model Found, Initializing Variables Randomly")
                self.session.run(tf.initialize_all_variables())
            self.loaded = True

    def save(self):
        if self.loaded:
            with self.session.as_default(), self.graph.as_default():
                saver = tf.train.Saver()
                saver.save(self.session, "agents/models/" + self.name + ".cpkt")
            print(self.name + " Save Successful")

    def get_action(self, state):
        with self.session.as_default(), self.graph.as_default():
            return self.session.run(self.network, feed_dict={self.state_ph: [state]})[0]

    def train(self, states, rewards):
        with self.session.as_default(), self.graph.as_default():
            self.session.run(self.train_operation, feed_dict={
                self.state_ph: states,
                self.actions_ph: rewards})
