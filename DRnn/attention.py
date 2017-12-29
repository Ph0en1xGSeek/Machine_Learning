import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep
from sklearn.decomposition import PCA
from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

learning_rate = 0.01
max_samples = 50000
batch_size = 250
display_step = 20

#before sending to rnn the input will be map to hidden layer of n_hidden-D first, and output of rnn is n_hidden too
n_hidden = 1
n_classes = 2




class BiRNN(object):

    def __init__(self, n_input, n_hidden, n_classes):

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self._initialize_weights()
        self.x = tf.placeholder("float", [None, n_input])
        self.input = tf.reshape(self.x, [-1, n_input, 1])
        # x = tf.split(self.x, 1, axis=1)

        # self.lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(
        #     n_hidden,
        #     forget_bias = 1.0
        # )
        #
        # self.lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(
        #     n_hidden,
        #     forget_bias = 1.0
        # )

        self.gru_fw_cell = tf.contrib.rnn.GRUCell(
            n_hidden
        )

        self.gru_bw_cell = tf.contrib.rnn.GRUCell(
            n_hidden
        )

        self.outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            # self.lstm_fw_cell,
            # self.lstm_bw_cell,
            self.gru_fw_cell,
            self.gru_bw_cell,
            self.input,
            dtype=tf.float32
        )

        #only need the last one of outputs
        concat = tf.concat([tf.reshape(self.outputs[0], [-1, n_input*n_hidden]), tf.reshape(self.outputs[1], [-1, n_input*n_hidden])], 1)
        # concat = tf.maximum(tf.reshape(self.outputs[0], [-1, n_input * n_hidden]), tf.reshape(self.outputs[1], [-1, n_input * n_hidden]))
        self.mid = tf.nn.softplus(tf.matmul(concat, self.weights) + self.biases)
        self.pred = tf.nn.sigmoid(tf.matmul(self.mid, self.reconstruct_weights) + self.reconstruct_biases)

        # self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.pred, self.x), 2.0))
        self.cost = tf.reduce_sum(-tf.reduce_sum(tf.nn.softmax(self.x) * tf.log(tf.nn.softmax(self.pred)), reduction_indices=[1]))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        # self.writer = tf.summary.FileWriter('./graphs', self.sess.graph)
        self.sess.run(init)

    def _initialize_weights(self):
        # self.weights = tf.Variable(tf.random_normal([2 * self.n_hidden, self.n_classes]))
        self.weights = tf.Variable(xavier_init(2*self.n_input*self.n_hidden, self.n_classes))
        self.biases = tf.Variable(tf.zeros([self.n_classes], dtype=tf.float32))

        # self.reconstruct_weights = tf.Variable(tf.random_normal([self.n_classes, self.n_input]))
        self.reconstruct_weights = tf.Variable(xavier_init(self.n_classes, self.n_input))
        self.reconstruct_biases = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))

    def getMid(self, X):
        return self.sess.run(self.mid, feed_dict={self.x : X})

    def partial_fit(self, X):
        #get the value of cost and optimize it
        cost, opt = self.sess.run((self.cost, self.optimizer),
                             feed_dict = {self.x : X})
        return cost
    def getOutput(self, X):
        return self.sess.run(self.outputs, feed_dict={self.x:X})
    def getFw(self, X):
        return self.sess.run(self.fw, feed_dict={self.x:X})
    def getBw(self, X):
        return self.sess.run(self.bw, feed_dict={self.x:X})

def get_random_block_from_data(data, batch_size):
    # get a batch size of batch_size randomly
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index: (start_index + batch_size)]

def standard_scale(X_train, X_test):
    #标准化数据，均值为0，标准差为1
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(x, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(pred, x), 2.0))

def main(_):

    dataset = 'Im'
    X_test = X_train = np.loadtxt("data/"+dataset+"_X.txt");
    n, m = X_test.shape
    labels = np.loadtxt("data/"+dataset+"_labels.txt");
    pca2 = PCA(n_components=2)
    Y_pca = pca2.fit_transform(X_test)

    X_train, X_test = standard_scale(X_train, X_test)
    pred = BiRNN(m, n_hidden, n_classes)
    step = 1
    tot_cost = 0
    while step * batch_size < max_samples:
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        # print(np.shape(batch_x))
        # batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = pred.partial_fit(batch_xs)
        tot_cost += cost / display_step
        if step % display_step == 0:
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(tot_cost))
            tot_cost = 0
        step += 1
    print("Optimization Finished")
    print("output", pred.getOutput(X_test)[-1].shape)

    # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))

    print('shape', labels.shape)
    Y = pred.getMid(X_test)
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(dataset)
    ax = fig.add_subplot(1, 2, 1)
    plt.title('attention-autoencoder')
    plt.scatter(Y[:, 0], Y[:, 1], c=labels, cmap=plt.cm.Spectral, marker='.')
    ax = fig.add_subplot(1, 2, 2)
    plt.title('pca')
    plt.scatter(Y_pca[:, 0], Y_pca[:, 1], c=labels, cmap=plt.cm.Spectral, marker='.')
    # pred.writer.close()
    plt.show()

if __name__ == '__main__':
    tf.app.run(main = main)
