# ML programs using TensorFlow, scikit learn, ...

------------------------------------------------------------------------------------
# What is Tensor? 
A tensor is a generalization of vectors and matrices to potentially higher dimensions. Internally, TensorFlow represents tensors as n-dimensional arrays of base datatypes. 
TensorFlow programs work by first building a graph of tf.Tensor objects, detailing how each tensor is computed based on the other available tensors and then by running parts of this graph to achieve the desired results. 

Main tensors: 
    tf.Variable
    tf.constant
    tf.placeholder
    tf.SparseTensor

Ref - https://www.tensorflow.org/programmers_guide/tensors 
Tensorflow Intro - https://www.youtube.com/watch?v=uh2Fh6df7Lg 

------------------------------------------------------------------------------------
Example: Hello world!
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
sess.close()


Example: Helloworld without session
import sys
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

a = tf.constant(2)
b = tf.constant(3)
c = a + b
print("c: %i" %c)

------------------------------------------------------------------------------------
# Variables 
A variable maintains state in the graph across calls to run(). You add a variable to the graph by constructing an instance of the class Variable.
The Variable() constructor requires an initial value for the variable, which can be a Tensor of any type and shape. 

https://www.tensorflow.org/api_docs/python/tf/Variable 

Example: 
import tensorflow as tf

x = tf.Variable(0)		# declares & initialize x to 0
y = tf.assign(x, 1)		# assigns 1 to 'x' and same to 'y' https://www.tensorflow.org/api_docs/python/tf/assign

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print sess.run(x)
print sess.run(y)
print sess.run(x)

"""
Output:
0
1
1
"""

------------------------------------------------------------------------------------
# Constants 
Creates a constant tensor.
The resulting tensor is populated with values of type dtype, as specified by arguments value and (optionally) shape. 

The argument value can be a constant value, or a list of values of type dtype. If value is a list, then the length of the list must be less than or equal to the number of elements implied by the shape argument (if specified). In the case where the list length is less than the number of elements specified by shape, the last element in the list will be used to fill the remaining entries.

https://www.tensorflow.org/api_docs/python/tf/constant 
https://learningtensorflow.com/lesson2/ 

Example:
import tensorflow as tf

x = tf.constant(35, name='x')
y = tf.Variable(x + 5, name='y')

# Constant 1-D Tensor populated with value list.
OneDlist = tf.constant([1, 2, 3, 4, 5, 6, 7])

# Constant 2-D tensor populated with scalar value -1.
TwoDlist = tf.constant(-1.0, shape=[2, 3])

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    print(session.run(y))              # prints 40
    print(session.run(OneDlist))       # prints [1 2 3 4 5 6 7]
    print(session.run(TwoDlist))       # prints [[-1. -1. -1.]
                                       #         [-1. -1. -1.]]


"""
Output:
40
[1 2 3 4 5 6 7]
[[-1. -1. -1.]
 [-1. -1. -1.]]
"""

------------------------------------------------------------------------------------
# Placeholders 
A placeholder is simply a variable that we will assign data to at a later date. It allows us to create our operations and build our computation graph, without needing the data. In TensorFlow terminology, we then feed data into the graph through these placeholders.

tf.placeholder(
    dtype,
    shape=None,
    name=None
)

Args:
    dtype: The type of elements in the tensor to be fed.
    shape: The shape of the tensor to be fed (optional). If the shape is not specified, you can feed a tensor of any shape.
    name: A name for the operation (optional).

Returns:
A Tensor that may be used as a handle for feeding a value, but not evaluated directly.

https://www.tensorflow.org/api_docs/python/tf/placeholder 

--- arithmetic_opeartion.py ---
#!/usr/bin/python

import tensorflow as tf

# build computational graph
a = tf.placeholder(tf.int64)
b = tf.placeholder(tf.int64)

addition = tf.add(a, b)
subtraction = tf.subtract(a, b)
multiply = tf.multiply(a, b)
divide = tf.divide(a, b)

# initialize variables
init = tf.global_variables_initializer()

# create session and run graph
with tf.Session() as sess:
    sess.run(init)
    print "addition: %d" % sess.run(addition, feed_dict={a: 3, b: 2})
    print "subraction: %d" % sess.run(subtraction, feed_dict={a: 3, b: 2})
    print "multiply: %d" % sess.run(multiply, feed_dict={a: 3, b: 2})
    print "divide: %d" % sess.run(divide, feed_dict={a: 3, b: 2})

# close session
sess.close()

------------------------------------------------------------------------------------
# How tensors are visualised in graph 

Refer "The basic script" topic in https://learningtensorflow.com/Visualisation/ 

------------------------------------------------------------------------------------
# NumPy 
NumPy is the fundamental package for scientific computing with Python. It contains among other things:
    a powerful N-dimensional array object
    sophisticated (broadcasting) functions
    tools for integrating C/C++ and Fortran code
    useful linear algebra, Fourier transform, and random number capabilities

NumPy should be used for larger lists/arrays of numbers, as it is significantly more memory efficient and faster to compute on than lists. It also provides a significant number of functions (such as computing the mean) that aren’t normally available to lists.

http://www.numpy.org/ 

------------------------------------------------------------------------------------
# Linear regression 
 Linear regression is a statistical model that examines the linear relationship between two (Simple Linear Regression ) or more (Multiple Linear Regression) variables - a dependent variable and independent variable(s). Linear relationship basically means that when one (or more) independent variables increases (or decreases), the dependent variable increases (or decreases) too

Ref : https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9 
Program - linear_regression.py, Linear_regression_sklearn.py 

------------------------------------------------------------------------------------
# Linear regression with multivariant 
 Multiple features

california_housing_train.csv
"longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income","median_house_value"
-114.310000,34.190000,15.000000,5612.000000,1283.000000,1015.000000,472.000000,1.493600,66900.000000
-114.470000,34.400000,19.000000,7650.000000,1901.000000,1129.000000,463.000000,1.820000,80100.000000
-114.560000,33.690000,17.000000,720.000000,174.000000,333.000000,117.000000,1.650900,85700.000000
-114.570000,33.640000,14.000000,1501.000000,337.000000,515.000000,226.000000,3.191700,73400.000000

Ref - https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9 
      https://www.youtube.com/watch?v=kllogfBujLs 
Program - linear_regression_multivariant.py, Linear_regression_sklearn_csv.py 

------------------------------------------------------------------------------------
# Logistic regression 

Logistic regression predicts the probability of an outcome that can only have two values (i.e. a dichotomy). 
Logistic Regression is used when the dependent variable(target) is categorical. 

For example, 
- To predict whether an email is spam (1) or (0)
- Whether the tumor is malignant (1) or not (0)

Explanation: 
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
=> (x * W) + b
=> ((1 x 784) * (784 x 10)) + (1 x 10)
=> (1 x 10) + (1 x 10)
=> (1 x 10)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
tf.log(pred) -> First, tf.log computes the logarithm of each element of pred. 
(y*tf.log(pred) -> Next, we multiply each element of y with the corresponding element of tf.log(pred). 
tf.reduce_sum(y*tf.log(pred) -> Then tf.reduce_sum adds the elements in the second dimension of pred, due to the reduction_indices=[1] parameter. 
	y = (1 x 10)
	batch size = 100
	result batch = (1 x 100), reduction_indices=[1] implies summing each element of vector in batch of 100
	e.g. one prediction vector will be like this
	[6.99726615e-06 + 1.82647607e-04 + 9.94143784e-01 + 4.29605879e-03 + 1.90415940e-05 + 5.07002114e-05 + 1.90451276e-04 + 7.19771407e-08 + 1.09627761e-03 + 1.39646618e-05] = 0.999999994993
tf.reduce_mean() -> Finally, tf.reduce_mean computes the mean over all the examples in the batch.


Output pred vs label:
 [2.81254568e-11 9.95265177e-13 2.02638359e-04 8.97943508e-04
  1.21668552e-06 9.40459373e-04 3.04663450e-09 6.78048934e-07
  9.96642828e-01 1.31431560e-03]								===> [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.] = 8 (high density value in vector)
[1.08923472e-03 1.72849496e-07 5.93256671e-03 3.49515933e-04
  6.75066784e-02 2.29581189e-03 6.28186448e-04 5.75977750e-02
  3.00991908e-02 8.34500849e-01]								===> [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.] = 9 (high density value in vector)

>>> print("Actual : Prediction")
Actual : Prediction
>>> for a_vec, p_vec in zip(actual_values, predicted_values):
...     print("%d : %d" % (a_vec.index(max(a_vec)), p_vec.index(max(p_vec))))
...
8 : 8
9 : 9


Ref - https://www.tensorflow.org/versions/r1.3/get_started/mnist/beginners 
Program - logistic_regression.py, Logistic_regression_sklearn_csv.py 

------------------------------------------------------------------------------------
