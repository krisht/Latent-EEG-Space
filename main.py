import os

import tensorflow as tf
import tensorflow.contrib.slim as slim


def brain_net(inputs):
    net = slim.layers.fully_connected(inputs, num_outputs = 500)

