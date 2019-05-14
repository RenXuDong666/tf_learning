from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug

def add_layer(input_data, input_size, output_size, active=None, dropout=True):
    Weight = tf.Variable(tf.random_normal([input_size, output_size]))
    bias = tf.Variable(tf.zeros([1, output_size])+0.1)
    Wx_b = tf.matmul(input_data, Weight)+bias
    if dropout:
        Wx_b = tf.nn.dropout(Wx_b, keep_prob)
    else:
        pass

    if active == None:
        output = Wx_b
    else:
        output = active(Wx_b)
    return output

def get_weight(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

def get_bais(shape, name):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)

def compute_accuracy(vx, vy):
    global prediction
    pre = sess.run(prediction, feed_dict={x_data:vx, ys:vy, keep_prob:1})
    accu = tf.equal(tf.argmax(pre, 1), tf.argmax(vy, 1))
    accu = tf.cast(accu, np.float32)
    res = sess.run(tf.reduce_mean(accu), feed_dict={x_data:vx, ys:vy, keep_prob:1})
    return res

mnist = input_data.read_data_sets("./MNIST_data",one_hot=True)
x_data = tf.placeholder(tf.float32,[None, 784])
xs = tf.reshape(x_data, [-1, 28, 28, 1])
ys = tf.placeholder(tf.float32,[None, 10])
keep_prob = tf.placeholder(tf.float32)

# cnn
weight_1 = get_weight([5, 5, 1, 32], "layer1_weight")
bias_1 = get_weight([32], "layer1_bias")
layer_1 = tf.nn.relu(tf.nn.conv2d(xs, weight_1, [1,1,1,1], "SAME")+bias_1)
layer_1 = tf.nn.max_pool(layer_1, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding="SAME")
# cnn
weight_2 = get_weight([5, 5, 32, 64], "later2_weight")
bias_2 = get_weight([64], "layer2_bias")
layer_2 = tf.nn.relu(tf.nn.conv2d(layer_1, weight_2, [1,1,1,1], "SAME")+bias_2)
layer_2 = tf.nn.max_pool(layer_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
# dnn
input_3 = tf.reshape(layer_2, [-1,49*64])
layer_3 = add_layer(input_3, 49*64, 1024, active = tf.nn.relu, dropout=True)
# dnn
prediction = add_layer(layer_3, 1024, 10, dropout=False)

# cross_entry = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(tf.clip_by_value(tf.nn.softmax(prediction), 1e-10, 1.0)),1))
# cross_entry_1 = -tf.reduce_sum(ys*tf.log(tf.clip_by_value(prediction, 1e-10, 1.0 )), name="cross")
cross_entry = tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=prediction)
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entry)
initial = tf.global_variables_initializer()

sess = tf.Session()
sess.run(initial)
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
for i in range(20000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train, feed_dict={x_data:batch_xs, ys:batch_ys, keep_prob:0.5})
    #print("by hand")
    #print(sess.run(cross_entry_1, feed_dict={x_data:batch_xs, ys:batch_ys, keep_prob:0.5}))
    #print("by tf")
    #print(sess.run(cross_entry_2, feed_dict={x_data:batch_xs, ys:batch_ys, keep_prob:0.5}))
    if i%50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
