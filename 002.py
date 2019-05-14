import numpy as np
import tensorflow as tf

one = tf.constant([1])
var = tf.Variable([1])

new = tf.add(one,var)
renew = tf.assign(var, new)
initial = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(initial)
    for _ in range(3):
        sess.run(renew)
        print(sess.run(var))
