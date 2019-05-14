import tensorflow as tf
import numpy as np

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

com = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(com, feed_dict={input1:[1.], input2:[2.]}))
