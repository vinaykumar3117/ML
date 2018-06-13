# ML programs using TensorFlow, scikit learn, ...

------------------------------------------------------------------------------------
What is Tensor?
A tensor is a generalization of vectors and matrices to potentially higher dimensions. Internally, TensorFlow represents tensors as n-dimensional arrays of base datatypes. 
TensorFlow programs work by first building a graph of tf.Tensor objects, detailing how each tensor is computed based on the other available tensors and then by running parts of this graph to achieve the desired results. 

Main tensors: 
    tf.Variable
    tf.constant
    tf.placeholder
    tf.SparseTensor

https://www.tensorflow.org/programmers_guide/tensors
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
Variables
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
Constants
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
Placeholders
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
How tensors are visualised in graph

Refer "The basic script" topic in https://learningtensorflow.com/Visualisation/

------------------------------------------------------------------------------------
NumPy
NumPy is the fundamental package for scientific computing with Python. It contains among other things:
    a powerful N-dimensional array object
    sophisticated (broadcasting) functions
    tools for integrating C/C++ and Fortran code
    useful linear algebra, Fourier transform, and random number capabilities

NumPy should be used for larger lists/arrays of numbers, as it is significantly more memory efficient and faster to compute on than lists. It also provides a significant number of functions (such as computing the mean) that aren’t normally available to lists.

http://www.numpy.org/

------------------------------------------------------------------------------------
Linear regression
 Linear regression is a statistical model that examines the linear relationship between two (Simple Linear Regression ) or more (Multiple Linear Regression) variables - a dependent variable and independent variable(s). Linear relationship basically means that when one (or more) independent variables increases (or decreases), the dependent variable increases (or decreases) too

Ref : https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9
Program - linear_regression.py

------------------------------------------------------------------------------------
Linear regression with multivariant
 Multiple features

california_housing_train.csv
"longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income","median_house_value"
-114.310000,34.190000,15.000000,5612.000000,1283.000000,1015.000000,472.000000,1.493600,66900.000000
-114.470000,34.400000,19.000000,7650.000000,1901.000000,1129.000000,463.000000,1.820000,80100.000000
-114.560000,33.690000,17.000000,720.000000,174.000000,333.000000,117.000000,1.650900,85700.000000
-114.570000,33.640000,14.000000,1501.000000,337.000000,515.000000,226.000000,3.191700,73400.000000

Ref - https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9
      https://www.youtube.com/watch?v=kllogfBujLs
Program - linear_regression_multivariant.py

------------------------------------------------------------------------------------
Logistic regression


