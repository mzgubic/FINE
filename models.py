import tensorflow as tf
from tensorflow.contrib import layers

def T(x):

    # need to have shared variables here!
    # only one hidden layer
    hidden = layers.relu(x, 10)
    
    # output
    output = layers.linear(hidden, 1)
    
    return output

def FINE(x_theta, x_theta_prime):
    
    # compute Ts
    T_theta = T(x_theta)
    T_theta_prime = T(x_theta_prime)
    
    # compute the loss
    FI = tf.reduce_mean(T_theta) - tf.math.log(tf.reduce_mean(tf.math.exp(T_theta_prime)))
    loss = -FI # FI = supremum
    
    # optimiser
    opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
    
    return FI, opt
