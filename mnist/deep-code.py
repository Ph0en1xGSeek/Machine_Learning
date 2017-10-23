from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

sess = tf.InteractiveSession()

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


#return a tensor that is normal distributed and has a shape of 'shape'
def weight_varible(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.2, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def deepnn(x):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    with tf.name_scope("conv1"):
        # filter 1 layer per time and 32 times totally
        W_conv1 = weight_varible([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        #[-1, 28, 28, 32]

    with tf.name_scope("pool1"):
        h_pool1 = max_pool_2x2(h_conv1)
        #[-1, 14, 14, 32]

    with tf.name_scope('conv2'):
        #filter 32 layers per time and 64 times totally
        W_conv2 = weight_varible([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        # [-1, 14, 14, 64]

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)
        # [-1, 7, 7, 64]

    with tf.name_scope('fc1'):
        W_fc1 = weight_varible([7*7*64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        #[1024]

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weight_varible([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        #[10]

    return y_conv, keep_prob

def main(_):

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])


    y_conv, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    with tf.name_scope('adam_optimizater'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(5000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
                print('step %d, test accuracy is %s'%(i, train_accuracy))
            train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})




if __name__ == "__main__":
    tf.app.run(main=main)