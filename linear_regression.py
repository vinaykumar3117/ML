import tensorflow as tf
import numpy as np

# For the values of input and output train data we need to have them in numpy array.
train_X = np.asarray([3.3, 4.4, 5.5, 6.71])
train_Y = np.asarray([1.7, 2.76, 2.09, 3.19])

X = tf.placeholder("float")
Y = tf.placeholder("float")

epoch = 1000
learning_rate = 0.1

# Lets initialize the value of W and b to some random values. Even you can give the value as zero.
W = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())

size = train_X.size

#Session encapsulates the environment in which operation objects are executed and evaluated.
sess = tf.Session()

#To Initialize the variable in the tensorflow context/session.
init = tf.global_variables_initializer()
sess.run(init)

# To check what is the value of W(weight) and b(bias)
#print sess.run(W)
#print sess.run(b)

#prediction = W * value_in_TrainX + b
pred = tf.add(tf.multiply(W, X), b)

# {SUM[(prediction - value_in_TrainY)^2]} / 2* input_array_size_X
cost = tf.reduce_sum(tf.square(pred - Y))/(2*size)

#Gradient decent algorithm to minimize cost.
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Train data in epoch number of iterations.
for i in range(epoch):
    # Need to extract each set of values from input and output(Train_X and Train_Y) arrays and Feed this to X and Y.
    for (x,y) in zip(train_X, train_Y):
        # Run the optimizer which in turn calculates cost and minimizes the cost using Gradient decent algorithm.
        sess.run(optimizer,feed_dict={X:x,Y:y})
    if (i + 1) % 100 == 0:
        # For every 100th iteration, print the cost value.
        print "Value of cost prediction at %d is :" % (i+1), sess.run(cost, feed_dict={X: train_X, Y: train_Y})


#Final minimized cost value, weights and bias.
print "Final value of cost is : ", sess.run(cost, feed_dict={X:train_X, Y:train_Y})
print "Final value of W is : ",sess.run(W)
print "Final value of b is : ",sess.run(b)

# close the session.
sess.close()


"""
Output:

Value of cost prediction at 100 is : 0.08493525
Value of cost prediction at 200 is : 0.0850673
Value of cost prediction at 300 is : 0.08525485
Value of cost prediction at 400 is : 0.08537343
Value of cost prediction at 500 is : 0.08543698
Value of cost prediction at 600 is : 0.08546899
Value of cost prediction at 700 is : 0.08548465
Value of cost prediction at 800 is : 0.08549231
Value of cost prediction at 900 is : 0.08549598
Value of cost prediction at 1000 is : 0.08549777
Final value of cost is :  0.08549777
Final value of W is :  0.4467973
Final value of b is :  0.258191
"""
