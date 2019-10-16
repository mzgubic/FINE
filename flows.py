import tensorflow as tf

# this class contains definitions for different normalising flows
class RadialFlow:

    def __init__(self, alpha, beta, name = "flow_trafo"):
        with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
            self.name = name
            self.alpha = tf.math.softplus(alpha) - 1.0
            self.beta = beta

    def forward(self, z):
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            return z + self.alpha * (z - self.beta)

    def backward(self, z):
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            return (z + self.alpha * self.beta) / (1.0 + self.alpha)

    def forward_derivative(self, z):
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            return tf.ones_like(z) * (1.0 + self.alpha)


class TombsFlow:

    def __init__(self, alpha, beta, name = "flow_trafo"):
        with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
            self.name = name
            self.alpha = tf.math.softplus(alpha) + 1e-6
            self.beta = tf.math.softplus(beta) + 1e-6

    def forward(self, z):
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            return self.alpha * tf.math.sinh(z * self.beta)

    def backward(self, z):
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            return tf.math.asinh(z / self.alpha) / self.beta

    def forward_derivative(self, z):
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            return self.alpha * self.beta * tf.math.cosh(z * self.beta)
