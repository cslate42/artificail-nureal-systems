#Google's advanced MNIST tutorial

import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
import tensorflow as tf

#### Begin TensorFlow Graph

#Placeholders for the input images and target output classes
x = tf.placeholder(tf.float32, shape=[None, 784]) #Any x 784 matrix
y_ = tf.placeholder(tf.float32, shape=[None, 10]) #Digit class for each mnist image

#Variable used and modified by the computation. Model parameters are usually these
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#Implement regression model
y = tf.nn.softmax(tf.matmul(x,W) + b)

#Cost function - 'reduce' means parallel sum/mean?
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#Automatic gradient descent!
#It does: compute gradients, compute parameter update steps, apply update steps
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#Calculate percentage accurage
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#### End TensorFlow Graph

#Start the session
with tf.Session() as sess:
    sess.run(init)

    #For each training iteration, load 50 training examples.
    #   Then run train_step, using feed_dict to replace the placeholder tensors x and y_
    #   with the training examples. You can replace any tensor with feed_dict
    for i in range(1000):
        batch = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
