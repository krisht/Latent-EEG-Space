# Krishna Thiyagarajan
# 02/20/2017
# Convolutional ANN for MNIST
# Computational Graphs for Machine Learning
# Prof. Chris Curro
#

import warnings
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim
import random
import numpy as np
warnings.filterwarnings('ignore')


class ConvANN:
    def __init__(self, sess, weight_dim, bias_dim, iterations,
                 batch_size, learn_rate, gamma, display_steps=100, num_in=num_inputs,
                 num_embed=128):

        self.sess = sess
        self.num_inputs = num_in
        self.num_embeds = num_embed
        self.weight_dims = weight_dim
        self.bias_dims = bias_dim
        self.iterations = iterations
        self.batch_size = batch_size
        self.display_steps = display_steps
        self.learn_rate = learn_rate
        self.gamma = gamma
        self.x = tf.placeholder(tf.float32, [None, 71, 125, 1])
        self.y = tf.placeholder(tf.float32, [None, self.num_embeds])
        self.dropout = tf.placeholder(tf.float32)
        self.build_model()

    def build_model(self):
        x = tf.reshape(self.x, shape=[-1, 71, 125, 1])

        self.yhat =slim.layers.conv2d(x, 32, 5, scope='conv1',trainable=True)



        self.yhat = self.conv2d(x, self.weight_dims['w1'], self.bias_dims['b1'])
        self.yhat = self.maxpool2d(self.yhat)
        self.yhat = self.conv2d(self.yhat, self.weight_dims['w2'],
                                self.bias_dims['b2'])
        self.yhat = self.maxpool2d(self.yhat)

        self.yhat = tf.reshape(self.yhat, [-1, self.weight_dims['w3'].get_shape().as_list()[0]])
        self.yhat = tf.add(tf.matmul(self.yhat, self.weight_dims['w3']), self.bias_dims['b3'])
        self.yhat = tf.nn.relu(self.yhat)

        self.yhat = tf.nn.dropout(self.yhat, self.dropout)

        self.yhat = tf.add(tf.matmul(self.yhat, self.weight_dims['w4']),
                           self.bias_dims['b4'])

        self.costs = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.yhat, labels=self.y))
        self.l2 = tf.reduce_sum(tf.get_collection('l2'))
        self.loss = self.costs + self.gamma * self.l2

        self.correct_pred = tf.equal(tf.argmax(self.yhat, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def train(self):
        model_vars = tf.get_collection('model_vars')
        self.optim = (tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.loss, var_list=model_vars))
        self.sess.run(tf.global_variables_initializer())

        for kk in range(self.iterations):
            batch_x, batch_y = mnist.train.next_batch(self.batch_size)
            self.sess.run([self.optim], feed_dict={self.x: batch_x,
                                                   self.y: batch_y, self.dropout: 0.5})
            if kk % 1 == 0:
                acc = self.sess.run(self.accuracy, feed_dict={self.x: mnist.validation.images[:5000],
                                                              self.y: mnist.validation.labels[:1000],
                                                              self.dropout: 1.0})
                print("Step: %d, Accuracy: %f" % (kk, acc))
        print("Optimization complete!")
        self.valid_accuracy()

    def valid_accuracy(self):
        acc = self.sess.run(self.accuracy, feed_dict={self.x: mnist.validation.images[:1000],
                                                      self.y: mnist.validation.labels[:1000], self.dropout: 1.0})
        print("Validation Accuracy: ", acc)

    def test_accuracy(self):
        acc = self.sess.run(self.accuracy, feed_dict={self.x: mnist.test.images[:500],
                                                      self.y: mnist.test.labels[:500], self.dropout: 1.0})
        print("Test Accuracy: ", acc)


# Run 1

sess_1 = tf.Session()

weight_dim_1 = {
    'w1': def_weight([5, 5, 1, 32], 'w11'),
    'w2': def_weight([5, 5, 32, 64], 'w12'),
    'w3': def_weight([7 * 7 * 64, 1024], 'w13'),
    'w4': def_weight([1024, num_embeds], 'w14')
}

bias_dim_1 = {
    'b1': def_bias([32], 'b11'),
    'b2': def_bias([64], 'b12'),
    'b3': def_bias([1024], 'b13'),
    'b4': def_bias([num_embeds], 'b14')
}

runs_1 = 1000
minibatch_1 = 1
learnRate_1 = 1e-3
gamma_1 = 1e-4

model_1 = ConvANN(sess_1, weight_dim_1, bias_dim_1,
                  runs_1, minibatch_1, learnRate_1, gamma_1)
model_1.train()
