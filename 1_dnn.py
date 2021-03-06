#this is for a random sequence
#
import tensorflow as tf
import numpy as np

x_data = np.random.rand(1000).astype(np.float32)
y_data = x_data*3+4

weights = tf.Variable(initial_value=tf.random_uniform([1],-1,1), trainable=True)
bias = tf.Variable(initial_value=tf.zeros([1]), trainable=True)

y = x_data * weights + bias
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step%20 == 0:
        print(step, sess.run(weights), sess.run(bias))




