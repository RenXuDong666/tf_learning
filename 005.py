from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

def add_layer(input_data, input_size, output_size, active=None):
    Weight = tf.Variable(tf.random_normal([input_size, output_size]))
    bias = tf.Variable(tf.zeros([1, output_size])+0.1)
    Wx_b = tf.matmul(input_data, Weight)+bias
    Wx_b = tf.nn.dropout(Wx_b, keep_prob)
    if active == None:
        output = Wx_b
    else:
        output = active(Wx_b)
    return output

def compute_accuracy(vx, vy):
    global prediction
    pre = sess.run(prediction, feed_dict={xs:vx, ys:vy, keep_prob:1})
    accu = tf.equal(tf.argmax(pre, 1), tf.argmax(vy, 1))
    accu = tf.cast(accu, np.float32)
    res = sess.run(tf.reduce_mean(accu))
    return res

mnist = input_data.read_data_sets("./MNIST_data",one_hot=True)
xs = tf.placeholder(tf.float32,[None, 784])
ys = tf.placeholder(tf.float32,[None, 10])
keep_prob = tf.placeholder(tf.float32)

# initial = tf.global_variables_initializer()
layer_1 = add_layer(xs, 784, 100, tf.nn.softmax)
prediction = add_layer(layer_1, 100, 10, tf.nn.softmax)
cross_entry = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),1))
train = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entry)
initial = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(initial)
    print("---")
    for i in range(20000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={xs:batch_xs, ys:batch_ys, keep_prob:0.5})
        if i%50 == 0:
            print(compute_accuracy(mnist.test.images, mnist.test.labels))


    
