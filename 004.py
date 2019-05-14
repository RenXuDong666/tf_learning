import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, active_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    bais = tf.Variable(tf.zeros([1, out_size])+0.1)

    w_x_b = tf.add(tf.matmul(inputs, Weights), bais)
    if active_function == None:
        outputs = w_x_b
    else:
        outputs = active_function(w_x_b)
    return outputs


x_data = np.linspace(-1,1,3000)[:,np.newaxis]
noise = np.random.normal(0,0.1,x_data.shape).astype(np.float32)
y_data = np.square(x_data)+0.5+noise

#draw picture
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion() #will not stop
plt.show()

xs = tf.placeholder(np.float32, [None, 1])
ys = tf.placeholder(np.float32, [None, 1])

#design net
l1 = add_layer(xs, 1, 10, tf.nn.relu)
l2 = add_layer(l1, 10, 1, None)

loss = tf.reduce_mean(np.square(ys-l2), 1)

train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
inital = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(inital)
    for i in range(1000):
        sess.run(train, feed_dict={xs:x_data, ys:y_data})
        if i%50 == 0:
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction = sess.run(loss, feed_dict={xs:x_data, ys:y_data})
            lines = ax.plot(x_data, prediction,'r-',lw=5)
            plt.pause(0.1)


