import tensorflow as tf

# this class contains definitions for different normalising flows
class RadialFlow:

    def __init__(self, alpha, beta, name = "flow_trafo"):
        with tf.variable_scope(name):
            self.alpha = alpha
            self.beta = beta

    def forward(self, z):
        return z + self.alpha * (z - self.beta)

    def backward(self, z):
        return (z + self.alpha * self.beta) / (1.0 + self.alpha)

    def forward_derivative(self, z):
        return 1.0 + self.alpha

    def backward_derivative(self, z):
        return 1.0 / self.forward_derivative(z)
