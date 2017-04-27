import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import random



class BrainNet:
    def __init__(self, input_shape=[None, 22, 71, 125], num_output=128, num_classes=6, restore_dir=None):
        self.sess = tf.Session()
        if restore_dir is not None:
            dir = tf.train.Saver()
            dir.restore(self.sess, restore_dir)
        else:
            self.build_model(input_shape, num_output)
        self.num_classes = num_classes

    def build_model(self, input_shape, num_output):
        self.inputs = tf.placeholder(tf.float32, shape=input_shape, name='input')
        self.num_output = num_output
        self.input_shape = input_shape
        self.keep_prob = tf.placeholder(tf.float32)


        with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected],
                           weights_initializer=tf.contrib.layers.xavier_initializer(seed=random.random(), uniform=True),
                           weights_regularizer=slim.l2_regularizer(1e-3)):
            print(self.inputs)
            net = slim.layers.conv2d(self.inputs, num_outputs=30, kernel_size=5, scope='conv1')
            print(net)
            net = slim.layers.max_pool2d(net, 3, scope='pool1')
            print(net)
            net = slim.layers.batch_norm(net)
            print(net)
            net = slim.layers.conv2d(net, num_outputs = 60, kernel_size=3, stride=1, scope='conv2')
            print(net)
            net = slim.layers.max_pool2d(net, 2, stride=1, scope='pool2')
            print(net)
            net = slim.layers.batch_norm(net)
            print(net)
            net = slim.layers.flatten(net, scope='flatten')
            print(net)
            net = slim.layers.fully_connected(net, 256)
            print(net)
            net = slim.layers.dropout(net, keep_prob = self.keep_prob)
            print(net)
            net = slim.layers.fully_connected(net, 128)
            print(net)
            net = slim.layers.fully_connected(net, num_output, activation_fn=None, weights_regularizer=None)
            print(net)
            self.net = net
            print(self.net)

    def train_model(self, learning_rate, keep_prob, train_data, batch_size, train_epoch, validation_data, validation_epoch, outdir=None):
        self.labels = tf.placeholder(tf.float32, shape=[None, self.num_classes])

        anchor_positive = 1
        anchor_negative = 1

        pred_loss = tf.reduce_sum(anchor_positive) + tf.reduce_sum(anchor_negative)

        weight_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
        total_loss = pred_loss + weight_loss;

        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        self.trainer = slim.learning.create_train_op(total_loss=total_loss, optimizer=optimizer)
        self.sess.run(tf.global_variables_initializer())

        data_tuple = self.reshape_data(train_data)

    def infer_embedding(self, data):
        data_tuple = self.reshape_data(data)
        batch_input = [x[0] for x in data_tuple]
        predictions = self.sess.run()
        return predictions



sample_input = tf.random_uniform([1, 22, 71, 125], dtype=tf.float32);


model = BrainNet()