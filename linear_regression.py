import tensorflow as tf
import numpy as np

train_X = np.asarray([3.3, 4.4, 5.5, 6.71])
train_Y = np.asarray([1.7, 2.76, 2.09, 3.19])

X = tf.placeholder("float")
Y = tf.placeholder("float")

epoch = 1000
learning_rate = 0.1

W = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())

size = train_X.size

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

print sess.run(W)
print sess.run(b)

pred = tf.add(tf.multiply(W, X), b)

cost = tf.reduce_sum(tf.square(pred - Y))/(2*size)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

for i in range(epoch):
    for (x,y) in zip(train_X, train_Y):
        sess.run(optimizer,feed_dict={X:x,Y:y})
    print sess.run(cost, feed_dict={X: train_X, Y: train_Y})

print sess.run(cost, feed_dict={X:train_X, Y:train_Y})
print sess.run(W)
print sess.run(b)
