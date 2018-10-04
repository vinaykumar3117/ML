import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing

data = pd.read_csv('california_housing_train.csv')

#print data.dtypes
#print data.head()
#print data['total_bedrooms']

# Data is dataframe here, need to convert this object to list.
var1 = data['total_bedrooms'].tolist()
var2 = data['households'].tolist()
y = data['median_income'].tolist()

size = data.shape[0]

X1 = tf.placeholder("float")
X2 = tf.placeholder("float")
Y = tf.placeholder("float")

W1 = tf.Variable([3], dtype=np.float32)
W2 = tf.Variable([3], dtype=np.float32)
b = tf.Variable([-3], dtype=np.float32)

epoch = 1000
learning_rate = 0.000001

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

pred = W1*X1 + W2*X2 + b

#pred = tf.add(tf.add(tf.multiply(W1, X1), tf.multiply(W2, X2)), b)

cost = tf.reduce_sum(tf.square(pred - Y))/(2*size)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

for i in range(epoch):
    # We can also pass individual values of var1, var2 and y  by extracting it as zip.
    sess.run(optimizer, {X1:var1, X2:var2, Y:y})
    #print sess.run(cost, {X1: var1, X2: var2, Y: y})

print "Final cost value: ", sess.run(cost, {X1: var1, X2: var2, Y: y})
print "W1: ", sess.run(W1)
print "W2: ", sess.run(W1)
print "b: ", sess.run(b)


"""
--------- Output ----------

Final cost value:  10.615431
W1:  [-0.00725167]
W2:  [-0.00725167]
b:  [-3.001202]

"""
