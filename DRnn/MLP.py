import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

n_1 = 100
n_2 = 50
n_3 = 25


def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, optimizer = tf.train.AdamOptimizer(), scale = 0.1):
        #n_input:输入变量数
        #n_hidden:隐藏层节点数
        #transfer_function:隐藏层激活函数
        #optimizer:优化器Adam
        #scale:高斯噪声系数

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_1 = n_1
        self.n_2 = n_2
        self.n_3 = n_3
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale

        # network structure
        # distribution on the hidden layer relies on the activation function
        self.x = tf.placeholder(tf.float32, [None, self.n_input], name='X')
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # encode
        # self.att1 = self.x * self.weights['attention1']
        self.layer1 = tf.nn.softplus(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        self.layer2 = tf.nn.softplus(tf.add(tf.matmul(self.layer1, self.weights['w2']), self.weights['b2']))
        self.layer3 = tf.nn.softplus(tf.add(tf.matmul(self.layer2, self.weights['w3']), self.weights['b3']))

        self.hidden = tf.nn.softplus(tf.add(tf.matmul(self.layer3, self.weights['w4']), self.weights['b4']))

        # decode
        self.layer5 = tf.nn.softplus(tf.add(tf.matmul(self.hidden, self.weights['w5']), self.weights['b5']))
        self.layer6 = tf.nn.softplus(tf.add(tf.matmul(self.layer5, self.weights['w6']), self.weights['b6']))
        self.layer7 = tf.nn.softplus(tf.add(tf.matmul(self.layer6, self.weights['w7']), self.weights['b7']))

        self.reconstruction = tf.nn.tanh(tf.add(tf.matmul(self.layer7, self.weights['w8']), self.weights['b8']))
        # self.reconstruction = (self.att2 * (1.0/self.weights['attention2']))

        # self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.cost = tf.reduce_sum(-tf.reduce_sum(tf.nn.softmax(self.x) * tf.log(tf.nn.softmax(self.reconstruction)), reduction_indices=[1]))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()

        self.sess.run(init)
        self.writer = tf.summary.FileWriter('./graphs', self.sess.graph)

    def _initialize_weights(self):
        all_weights = {}
        all_weights['attention1'] = (tf.Variable(tf.ones([self.n_input], dtype=tf.float32), name='attention1'))
        # all_weights['attention2'] = (tf.Variable(tf.ones([self.n_input], dtype=tf.float32)))

        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_1), name='w1')
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_1], dtype=tf.float32), name='b1')
        all_weights['w2'] = tf.Variable(xavier_init(self.n_1, self.n_2), name='w2')
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_2], dtype=tf.float32), name='b2')
        all_weights['w3'] = tf.Variable(xavier_init(self.n_2, self.n_3), name='w3')
        all_weights['b3'] = tf.Variable(tf.zeros([self.n_3], dtype=tf.float32), name='b3')
        all_weights['w4'] = tf.Variable(xavier_init(self.n_3, self.n_hidden), name='w4')
        all_weights['b4'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32), name='b4')

        all_weights['w5'] = tf.Variable(xavier_init(self.n_hidden, self.n_3), name='w5')
        all_weights['b5'] = tf.Variable(tf.zeros([self.n_3], dtype=tf.float32), name='b5')
        all_weights['w6'] = tf.Variable(xavier_init(self.n_3, self.n_2), name='w6')
        all_weights['b6'] = tf.Variable(tf.zeros([self.n_2], dtype=tf.float32), name='b6')
        all_weights['w7'] = tf.Variable(xavier_init(self.n_2, self.n_1), name='w7')
        all_weights['b7'] = tf.Variable(tf.zeros([self.n_1], dtype=tf.float32), name='b7')
        all_weights['w8'] = tf.Variable(xavier_init(self.n_1, self.n_input), name='w8')
        all_weights['b8'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32), name='b8')
        return all_weights

    def partial_fit(self, X):
        #get the value of cost and optimize it
        cost, opt = self.sess.run((self.cost, self.optimizer),
                             feed_dict = {self.x : X, self.scale : self.training_scale})
        return cost

    def calc_total_cost(self, X):
        #get the value of cost
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale : self.training_scale})

    def transform(self, X):
        #get the value of hidden layer
        return self.sess.run(self.hidden, feed_dict={self.x:X, self.scale:self.training_scale})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size =  self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden : hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x : X, self.scale : self.training_scale})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])

    def getAttent1(self):
        return self.sess.run(self.weights['attention1'])

    # def getAttent2(self):
        # return self.sess.run(self.weights['attention2'])

def standard_scale(X_train, X_test):
    #标准化数据，均值为0，标准差为1
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

def get_random_block_from_data(data, batch_size):
    #get a batch size of batch_size randomly
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index : (start_index + batch_size)]

def main(_):
    # mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    # print (mnist.test.labels)
    # X_train = mnist.train.images
    # X_test = mnist.test.images
    # labels = mnist.test.labels

    dataset = 'mnist2500'
    X_test = X_train = np.loadtxt("data/"+dataset+"_X.txt");
    try:
        labels = np.loadtxt("data/"+dataset+"_labels.txt");
    except IOError:
        labels = np.zeros(X_test.shape[0])
    # print (labels)
    n, m = X_train.shape

    # pca = PCA(n_components=50)
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.fit_transform(X_test)

    pca2 = PCA(n_components=2)
    Y_pca = pca2.fit_transform(X_test)

    X_train, X_test = standard_scale(X_train, X_test)


    n_samples = X_train.shape[0]
    training_epochs = 100
    batch_size = n_samples // 10
    display_step = 1

    autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=m,
                                                   n_hidden=2,
                                                   optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                                   scale=0.005)
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples // batch_size)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)

            cost = autoencoder.partial_fit(batch_xs)
            avg_cost += cost / n_samples * batch_size

        if epoch % display_step == 0:
            print('Epoch:', '%04d'%(epoch+1), "cost=", '{:.9f}'.format(avg_cost))
    # print('Total cost: ' + str(autoencoder.calc_total_cost(X_test)))
    Y = autoencoder.transform(X_test)


    fig = plt.figure(figsize=(15, 8))
    plt.suptitle(dataset)
    ax = fig.add_subplot(1, 2, 1)
    plt.title('MLP')
    plt.scatter(Y[:,0], Y[:,1], c=labels, cmap=plt.cm.Spectral, marker='.')
    ax = fig.add_subplot(1, 2, 2)
    plt.title('pca')
    plt.scatter(Y_pca[:,0], Y_pca[:,1], c=labels, cmap=plt.cm.Spectral, marker='.')
    weight = autoencoder.getWeights()
    atten1 = autoencoder.getAttent1()
    atten1 = np.reshape(atten1, [1, m])
    # atten2 = autoencoder.getAttent2()
    # atten2 = np.reshape(atten2, [1, m])
    # fig2 = plt.figure(2)
    # plt.pcolor(np.abs(atten1),  # 指定绘图数据
    #            cmap=plt.cm.Blues,  # 指定填充色
    #            edgecolors='white'  # 指点单元格之间的边框色
    #            )
    # fig2 = plt.figure(3)
    # plt.pcolor(atten2,  # 指定绘图数据
    #            cmap=plt.cm.Blues,  # 指定填充色
    #            edgecolors='white'  # 指点单元格之间的边框色
    #            )
    autoencoder.writer.close()
    plt.show()

    fig, ax = plt.subplots()
    sc = plt.scatter(x, y)

    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):

        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}, {}".format(" ".join(list(map(str, ind["ind"]))),
                               " ".join([labels[n] for n in ind["ind"]]))
        annot.set_text(text)
        # annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        # annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.show()




if __name__ == '__main__':
    tf.app.run(main = main)
