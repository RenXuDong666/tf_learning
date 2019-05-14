import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 1e-4
hidden_layers = [256, 256]
lstm_num = 2
class_num = 10
batch_size = 32
sequence_len = 28

#data
mnist = input_data.read_data_sets("./MNIST_data", one_hot=True) 

x = tf.placeholder(dtype=tf.float32, shape=[None,28*28])
x_data = tf.reshape(x, [-1, 28, 28])
y_data = tf.placeholder(dtype=tf.float32, shape=[None, 10])

# RNN
lstms = [tf.nn.rnn_cell.LSTMCell(num_units=hidden_layer, forget_bias=1.0, state_is_tuple=True) for hidden_layer in hidden_layers]
multi_lstm = tf.nn.rnn_cell.MultiRNNCell(lstms)
initial_state = multi_lstm.zero_state(batch_size=32, dtype=tf.float32)
outputs, states = tf.nn.dynamic_rnn(multi_lstm, x_data, initial_state=initial_state, dtype=tf.float32, time_major=False)
final_state = outputs[:, -1, :]

#DNN
Weight = tf.Variable(tf.random_normal([hidden_layers[1], class_num], stddev=0.1))
bais = tf.Variable(tf.constant(0.1, shape=[class_num]))
result = tf.nn.softmax(tf.matmul(final_state, Weight)+bais)

#loss
loss = -tf.reduce_sum(tf.log(result)*y_data, 1)

#accu
accu = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(result, 1), tf.argmax(y_data, 1)), tf.float32))

#op
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
init_all_variable = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init_all_variable)
    for i in range(1000):
        _x, _y = mnist.train.next_batch(batch_size)
        sess.run(train_op, feed_dict={x: _x, y_data: _y})
        if i%50 == 0:
            test_x, test_y = mnist.test.next_batch(batch_size)
            print(sess.run(accu, feed_dict={x: test_x, y_data: test_y}))

