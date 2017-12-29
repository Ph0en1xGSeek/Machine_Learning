from distutils.version import  LooseVersion
import warnings
import tensorflow as tf
import numpy as np
from tensorflow.contrib import seq2seq


# Number of Epochs
num_epochs = 100
# Batch Size
batch_size = 64
# RNN Size
rnn_size = 512
# Embedding Dimension Size
embed_dim = 200
# Sequence Length
seq_length = 20
# Learning Rate
learning_rate = 0.01
# Show stats for every n number of batches
show_every_n_batches = 100

def load_data(fname):
    with open(fname, 'r', encoding="utf8") as f:
        text = f.read()

    data = text.split()
    return data

text = load_data('data/split.txt')
print('The first 10 words: {}'.format(text[:10]))

vocab = set(text)
vocab_to_int = {w: idx for idx, w in enumerate(vocab)}
int_to_vocab = {idx: w for idx, w in enumerate(vocab)}

print('Total words: {}'.format(len(text)))
print('Vocab size: {}'.format(len(vocab)))

# 文本转化为数字
int_text = [vocab_to_int[w] for w in text]


def get_input():
    '''
    input layers
    :return:
    '''
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    return inputs, targets, learning_rate

def get_init_cell(batch_size, rnn_size):
    '''
    堆叠 RNN cell
    :param batch_size:
    :param rnn_size: 隐藏层大小
    :return:
    '''
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    cell = tf.contrib.rnn.MultiRNNCell([lstm]) # 仅堆叠一层

    initial_state = cell.zero_state(batch_size, tf.float32)
    # 定义 name
    initial_state = tf.identity(initial_state, 'initial_state')
    return cell, initial_state

def get_embed(input_data, vocab_size, embed_dim):
    '''
    embedding from word -> embed vector
    :param input_data:
    :param vocab_size:
    :param embed_dim:
    :return:
    '''
    embedding = tf.Variable(tf.random_uniform([vocab_size, embed_dim], -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_data)

    return embed

def build_rnn(cell, inputs):
    '''
    rnn model
    :param cell:
    :param inputs:
    :return:
    '''
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

    final_state = tf.identity(final_state, 'final_state')
    return outputs, final_state

def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    '''
    build the nerual network
    :param cell:
    :param rnn_size:
    :param input_data:
    :param vocab_size:
    :param embed_dim:
    :return:
    '''
    embed = get_embed(input_data, vocab_size, embed_dim)
    outputs, final_state = build_rnn(cell, embed)

    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)

    return logits, final_state

def get_batches(int_text, batch_size, seq_length):
    batch = batch_size * seq_length
    n_batch = len(int_text) // batch
    int_text = np.array(int_text[:batch * n_batch])
    int_text_targets = np.zeros_like(int_text)
    int_text_targets[:-1], int_text_targets[-1] = int_text[1:], int_text[0]

    x = np.split(int_text.reshape(batch_size, -1), n_batch, -1)
    y = np.split(int_text_targets.reshape(batch_size, -1), n_batch, -1)

    return np.stack((x, y), axis=1)


train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_input()
    input_data_shape = tf.shape(input_text)

    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    probs = tf.nn.softmax(logits, name='probs')

    cost = seq2seq.sequence_loss(
        logits,
        targets,
        weights=tf.ones([input_data_shape[0], input_data_shape[1]])
    )

    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

batches = get_batches(int_text, batch_size, seq_length)

save_dir = './save'

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # 每训练一定阶段对结果进行打印
            if (epoch * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch,
                    batch_i,
                    len(batches),
                    train_loss))

    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Training Finished')

def get_tensors(loaded_graph):

    inputs = loaded_graph.get_tensor_by_name('inputs:0')
    initial_state = loaded_graph.get_tensor_by_name('initial_state:0')
    final_state = loaded_graph.get_tensor_by_name('final_state:0')
    probs = loaded_graph.get_tensor_by_name('probs:0')
    return inputs, initial_state, final_state, probs

def pick_word(probablities, int_to_vocab):

    result = np.random.choice(len(probablities), 50, p=probablities)
    return int_to_vocab[result[0]]

gen_length = 300

prime_word = '我爱上'

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    loader = tf.train.import_meta_graph(save_dir + '.meta')
    loader.restore(sess, save_dir)

    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    gen_sentences = [prime_word]
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    for n in range(gen_length):
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state}
        )

        pred_word = pick_word(probabilities[0, dyn_seq_length-1], int_to_vocab)

        gen_sentences.append(pred_word)

    lyrics = ' '.join(gen_sentences)
    lyrics = lyrics.replace(';', '\n')
    lyrics = lyrics.replace('.', ' ')
    lyrics = lyrics.replace(' ', '')

    print(lyrics)
