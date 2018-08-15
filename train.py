import tensorflow as tf
from preprocess import *
import numpy as np
win_unicode_console.enable()


def create(x, layer_sizes):
    next_layer_input = x
    encoding_matrices = []
    for dim in layer_sizes:
        input_dim = int(next_layer_input.get_shape()[1])
        W = tf.Variable(tf.random_uniform([input_dim, dim], -1.0 / math.sqrt(input_dim), 1.0 / math.sqrt(input_dim)))
        b = tf.Variable(tf.zeros([dim]))
        encoding_matrices.append(W)
        output = tf.nn.relu(tf.matmul(next_layer_input, W) + b)
        next_layer_input = output

    encoded_x = next_layer_input

    layer_sizes.reverse()
    encoding_matrices.reverse()

    for i, dim in enumerate(layer_sizes[1:] + [int(x.get_shape()[1])]):
        W = tf.transpose(encoding_matrices[i])
        b = tf.Variable(tf.zeros([dim]))
        output = tf.nn.relu(tf.matmul(next_layer_input, W) + b)
        next_layer_input = output

    reconstructed_x = next_layer_input

    return {
        'encoded': encoded_x,
        'decoded': reconstructed_x,
        'cost': tf.sqrt(tf.reduce_mean(tf.square(x - reconstructed_x)))
    }


def deep_test(adj,num_hidden,itr,ret):
    num_input = adj.shape[0]
    sess = tf.Session()
    x = tf.placeholder("float", [num_input,num_input])
    autoencoder = create(x, num_hidden)
    init = tf.initialize_all_variables()
    sess.run(init)
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(autoencoder['cost'])
    for i in range(itr):
        sess.run(train_step, feed_dict={x: adj})
        if i % 1000 == 0:
            print(i, " cost", sess.run(autoencoder['cost'], feed_dict={x: adj}))
        if i == itr-1 :
            nXF = sess.run(autoencoder['encoded'], feed_dict={x: adj})
            cost = sess.run(autoencoder['cost'], feed_dict={x: adj})
            if ret == 1:
                return  nXF
            if ret == 0:
                return cost
