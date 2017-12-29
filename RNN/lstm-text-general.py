import time
from collections import namedtuple

import numpy as np
import tensorflow as tf

batch_size = 100         # Sequences per batch
num_steps = 100          # Number of sequence steps per batch
lstm_size = 512         # Size of hidden layers in LSTMs
num_layers = 2          # Number of LSTM layers
learning_rate = 0.001    # Learning rate
keep_prob = 0.5         # Dropout keep probability


with open('data/anna.txt', 'r') as f:
    text = f.read()

vocab = set(text)
# print(len(vocab))
vocab_to_int = {c:i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

# print(text[:100])
# print(encoded[:100])

def get_batches(arr, n_seqs, n_steps):

    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)

    arr = arr[:batch_size * n_batches]

    arr = arr.reshape((n_seqs, -1))

    for n in range(0, arr.shape[1], n_steps):
        # [n_seqs, n_steps*n_batches]
        x = arr[:, n:n+n_steps]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y

batches = get_batches(encoded, 10, 50)
x, y = next(batches)
# print('x\n', x[:10, :10])
# print('\ny\n', y[:10, :10])

def build_inputs(num_seqs, num_steps):

    inputs = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='inputs')
    targets = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='targets')

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return inputs,targets, keep_prob

def build_lstm_dropout_cell(lstm_size, keep_prob):
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

def build_lstm(lstm_size, num_layers, batch_size, keep_prob):

    # lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    #
    # drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    # 新的tensorflow不能用同一个lstm实例构造lstm层了
    drop = build_lstm_dropout_cell

    cell = tf.contrib.rnn.MultiRNNCell([drop(lstm_size, keep_prob) for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)

    return cell, initial_state

def build_output(lstm_output, in_size, out_size):

    # 把所有时间的output连接起来(no need)
    # seq_output = tf.concat(lstm_output, axis=1)
    x = tf.reshape(lstm_output, [-1, in_size])

    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))

    logits = tf.matmul(x, softmax_w) + softmax_b
    out = tf.nn.softmax(logits, name='predictions')

    return out, logits

def build_loss(logits, targets, lstm_size, num_classes):
    y_one_hot = tf.one_hot(targets, num_classes)
    # 感觉不reshape也是一样的
    # y_reshaped = tf.reshape(y_one_hot, logits.get_shape())

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_one_hot)
    loss = tf.reduce_mean(loss)

    return loss

def build_optimizer(loss, learning_rate, grad_clip):

    #返回所有需要训练的变量列表
    tvars = tf.trainable_variables()
    #当梯度大于grad_clip时会把梯度的大小置为grad_clip
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))

    return optimizer

class CharRNN:

    def __init__(self, num_classes, batch_size=64, num_steps=50,
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False):
        '''

        :param num_classes: number of kinds of vocab len(vocab)
        :param batch_size:
        :param num_steps:
        :param lstm_size:
        :param num_layers:
        :param learning_rate:
        :param grad_clip:
        :param sampling:
        '''
        if sampling == True:
            #SGD ?
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()

        # Input Layer
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)

        # LSTM Layer
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)

        # one-hot encode the input
        #### 输入和输出都是以one-hot的形式
        #### 输入前要进行转换 把数字转换为onehot
        #### 输出的one-hot每一位代表这个数出现的概率
        x_one_hot = tf.one_hot(self.inputs, num_classes)

        # run RNN
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state

        # predict the result
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)

        # Loss and Optimizer
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)

# epochs = 5
#
# save_every_n = 200
#
#
# model = CharRNN(len(vocab), batch_size=batch_size, num_steps=num_steps,
#                 lstm_size=lstm_size, num_layers=num_layers, learning_rate=learning_rate)
#
# saver = tf.train.Saver(max_to_keep=100)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     counter = 0
#     for e in range(epochs):
#         new_state = sess.run(model.initial_state)
#         loss = 0
#         for x, y in get_batches(encoded, batch_size, num_steps):
#             # print(x)
#             counter += 1
#             start = time.time()
#             feed = {model.inputs: x,
#                     model.targets: y,
#                     model.keep_prob: keep_prob,
#                     model.initial_state: new_state}
#             batch_loss, new_state, _ = sess.run([model.loss, model.final_state, model.optimizer], feed_dict=feed)
#
#             end = time.time()
#             # control the print lines
#             if counter % 10 == 0:
#                 print('轮数: {}/{}... '.format(e + 1, epochs),
#                       '训练步数: {}... '.format(counter),
#                       '训练误差: {:.4f}... '.format(batch_loss),
#                       '{:.4f} sec/batch'.format((end - start)))
#
#             # if (counter % save_every_n == 0):
#             #     saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
#
#     saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))

def pick_top_n(preds, vocab_size, top_n=5):
    '''
    选取最可能的n个字符
    :param preds:
    :param vocab_size:
    :param top_n:
    :return:
    '''
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    # print(p)
    c= np.random.choice(vocab_size, 1, p=p)[0]
    return c

def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):

    samples = [c for c in prime]
    model = CharRNN(len(vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:

        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)

        for c in prime:
            x = np.zeros((1, 1))
            x[0, 0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            # 中间状态 new_state 得到保存
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)

        c = pick_top_n(preds, len(vocab))

        samples.append(int_to_vocab[c])

        for i in range(n_samples):
            x[0, 0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)

            c = pick_top_n(preds, len(vocab))

            samples.append(int_to_vocab[c])
    return ''.join(samples)

checkpoint = tf.train.latest_checkpoint('checkpoints')
print(checkpoint)
samp = sample(checkpoint, 500, lstm_size, len(vocab), prime="The ")
print(samp)


