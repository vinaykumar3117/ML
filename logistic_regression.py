import tensorflow as tf

# Import MNIST data - Already downloaded
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data", one_hot=True)

# Parameters
learning_rate = 0.7
training_epochs = 10000
batch_size = 100

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784 (images)
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes (labels)

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# Alternate way (gives less accuracy) - https://www.tensorflow.org/versions/r1.3/get_started/mnist/beginners
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    #print("mnist.train.num_examples: ", (int(mnist.train.num_examples)))

    # Training cycle
    for epoch in range(training_epochs):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
        # Display logs per epoch step
        if (epoch+1) % 1000 == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})))
            #print ("The predicted value is:", np.array(sess.run(pred, feed_dict={x: batch_xs, y: batch_ys})).tolist()) # here since the batch size is 100 it gets 100 predicted vectors.
            #print("The label is:", np.array(sess.run(y,feed_dict={y: batch_ys})).tolist()) # the actual image in the form 9 column vector like [0,0,1,0 ..]
            predicted_value_vector = np.array(sess.run(pred, feed_dict={x: batch_xs, y: batch_ys})).tolist()
            actual_label_vector = np.array(sess.run(y,feed_dict={y: batch_ys})).tolist()
            #print("actual value vector", actual_label_vector)
            #print("predicted_value_vector value vector", predicted_value_vector);
            for actual_vector, predicted_vector in zip(actual_label_vector, predicted_value_vector):
                print("Actual :-> %d :|: Predicted :-> %d" % (actual_vector.index(max(actual_vector)),predicted_vector.index(max(predicted_vector))))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


"""
Output:
Epoch: 1000 cost= 0.222599283
Actual :-> 8 :|: Predicted :-> 8
Actual :-> 5 :|: Predicted :-> 5
Actual :-> 8 :|: Predicted :-> 8
Actual :-> 0 :|: Predicted :-> 0
Actual :-> 5 :|: Predicted :-> 6
Actual :-> 2 :|: Predicted :-> 2
Actual :-> 8 :|: Predicted :-> 8
Actual :-> 6 :|: Predicted :-> 6
Actual :-> 1 :|: Predicted :-> 1
...
Optimization Finished!
Accuracy: 0.9197
"""

