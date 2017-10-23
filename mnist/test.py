from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import math



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# y = tf.nn.softmax(tf.matmul(x, W) + b)
y = tf.matmul(x, W) + b


y_ = tf.placeholder(tf.float32, [None, 10])
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#evaluaing value
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run() #Thanks to InteractiveSession

index = 0
pre_entropy = 0
MIN_ENTROPY = 1e-4
MAX_ENTROPY = 0.97

for _ in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(100) # each loop get 100 random data from set
    sess.run(train_step, {x: batch_xs, y_: batch_ys})
    index+=1
    cur_entropy = sess.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels})
    print("index #%d: current entropy is %s"%(index, cur_entropy))
    if math.fabs(cur_entropy - pre_entropy) < MIN_ENTROPY:
        break
    if(cur_entropy > MAX_ENTROPY):
        break

