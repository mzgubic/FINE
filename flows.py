import tensorflow as tf
import sys

# this class contains definitions for different normalising flows
class LinearRadialFlow:

    def __init__(self, alpha, beta, gamma, name = "flow_trafo"):
        with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
            self.name = name
            self.alpha = tf.math.exp(alpha) - 0.99
            self.beta = beta
            self.gamma = gamma

    def forward(self, z):
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            return z + self.alpha * (z - self.beta)

    def backward(self, z):
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            return (z + self.alpha * self.beta) / (1.0 + self.alpha)

    def forward_derivative(self, z):
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            return tf.ones_like(z) * (1.0 + self.alpha)

class RadialFlow:

    def __init__(self, alpha, beta, gamma, name = "flow_trafo"):
        with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
            self.name = name
            self.alpha = tf.math.exp(alpha)
            self.beta = tf.math.exp(beta) - 0.99
            self.gamma = gamma

    def forward(self, z):
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            return z + self.alpha * self.beta / (self.alpha + tf.math.abs(z - self.gamma)) * (z - self.gamma)

    def backward(self, z):
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            zero = tf.zeros_like(z)

            self.z_in = z
            
            # compute the values of the two different discriminants
            self.discriminant_A = 4 * z * (self.alpha - self.gamma) + 4 * self.alpha * self.beta * self.gamma + tf.math.square(z - self.alpha * (1 + self.beta) + self.gamma)
            self.discriminant_B = tf.math.square(z + self.alpha + self.alpha * self.beta + self.gamma) - 4 * (self.alpha * self.beta * self.gamma + z * (self.alpha + self.gamma))
            
            # compute the values of the four branches
            self.branch_A1 = 0.5 * (z - self.alpha - self.alpha * self.beta + self.gamma - tf.math.sqrt(self.discriminant_A))
            self.branch_A2 = 0.5 * (z - self.alpha - self.alpha * self.beta + self.gamma + tf.math.sqrt(self.discriminant_A))

            self.branch_B1 = 0.5 * (z + self.alpha + self.alpha * self.beta + self.gamma - tf.math.sqrt(self.discriminant_B))
            self.branch_B2 = 0.5 * (z + self.alpha + self.alpha * self.beta + self.gamma + tf.math.sqrt(self.discriminant_B))

            # boolean masks to determine which branch to take
            # Note: these will always be valid booleans, can never be NaN
            self.active_A1 = tf.math.logical_and(tf.greater(self.discriminant_A, 0), tf.greater(self.branch_A1, self.gamma))
            self.active_A2 = tf.math.logical_and(tf.greater(self.discriminant_A, 0), tf.greater(self.branch_A2, self.gamma))

            self.active_B1 = tf.math.logical_and(tf.greater(self.discriminant_B, 0), tf.less(self.branch_B1, self.gamma))
            self.active_B2 = tf.math.logical_and(tf.greater(self.discriminant_B, 0), tf.less(self.branch_B2, self.gamma))

            # -----
            # this is to work around a "feature" of gradient computations that shows up when one of the branches
            # in tf.where involves NaN values (even though they are never selected)
            # -----
            self.z_in_safeval_discriminant_A = -self.alpha + self.alpha * self.beta + self.gamma + tf.sqrt(1 + 4 * tf.square(self.alpha) * (tf.abs(self.beta) - self.beta))
            self.z_in_safe_A1 = tf.where(self.active_A1, self.z_in, self.z_in_safeval_discriminant_A * tf.ones_like(self.z_in))
            self.z_in_safe_A2 = tf.where(self.active_A2, self.z_in, self.z_in_safeval_discriminant_A * tf.ones_like(self.z_in))

            self.discriminant_safe_A1 = 4 * self.z_in_safe_A1 * (self.alpha - self.gamma) + 4 * self.alpha * self.beta * self.gamma + tf.math.square(self.z_in_safe_A1 - self.alpha * (1 + self.beta) + self.gamma)
            self.discriminant_safe_A2 = 4 * self.z_in_safe_A2 * (self.alpha - self.gamma) + 4 * self.alpha * self.beta * self.gamma + tf.math.square(self.z_in_safe_A2 - self.alpha * (1 + self.beta) + self.gamma)

            self.branch_safe_A1 = 0.5 * (self.z_in_safe_A1 - self.alpha - self.alpha * self.beta + self.gamma - tf.math.sqrt(self.discriminant_safe_A1))
            self.branch_safe_A2 = 0.5 * (self.z_in_safe_A2 - self.alpha - self.alpha * self.beta + self.gamma + tf.math.sqrt(self.discriminant_safe_A2))

            self.z_in_safeval_discriminant_B = self.alpha - self.alpha * self.beta + self.gamma + tf.sqrt(1 + 4 * tf.square(self.alpha) * (tf.abs(self.beta) - self.beta))
            self.z_in_safe_B1 = tf.where(self.active_B1, self.z_in, self.z_in_safeval_discriminant_B * tf.ones_like(self.z_in))
            self.z_in_safe_B2 = tf.where(self.active_B2, self.z_in, self.z_in_safeval_discriminant_B * tf.ones_like(self.z_in))

            self.discriminant_safe_B1 = tf.math.square(self.z_in_safe_B1 + self.alpha + self.alpha * self.beta + self.gamma) - 4 * (self.alpha * self.beta * self.gamma + self.z_in_safe_B1 * (self.alpha + self.gamma))
            self.discriminant_safe_B2 = tf.math.square(self.z_in_safe_B2 + self.alpha + self.alpha * self.beta + self.gamma) - 4 * (self.alpha * self.beta * self.gamma + self.z_in_safe_B2 * (self.alpha + self.gamma))

            self.branch_safe_B1 = 0.5 * (self.z_in_safe_B1 + self.alpha + self.alpha * self.beta + self.gamma - tf.math.sqrt(self.discriminant_safe_B1))
            self.branch_safe_B2 = 0.5 * (self.z_in_safe_B2 + self.alpha + self.alpha * self.beta + self.gamma + tf.math.sqrt(self.discriminant_safe_B2))            
            # -----
            
            # combine the different branches in their respective domains of validity
            self.selected_A1 = tf.where(self.active_A1, self.branch_safe_A1, zero)
            self.selected_A2 = tf.where(self.active_A2, self.branch_safe_A2, zero)

            self.selected_B1 = tf.where(self.active_B1, self.branch_safe_B1, zero)
            self.selected_B2 = tf.where(self.active_B2, self.branch_safe_B2, zero)

            return self.selected_A1 + self.selected_A2 + self.selected_B1 + self.selected_B2

    def forward_derivative(self, z):
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            return 1.0 + self.alpha * self.beta / (self.alpha + tf.math.abs(z - self.gamma)) - self.alpha * self.beta * (z - self.gamma) * tf.math.sign(z - self.gamma) / tf.math.square(self.alpha + tf.math.abs(z - self.gamma))

class TombsFlow:

    def __init__(self, alpha, beta, gamma, name = "flow_trafo"):
        with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
            self.name = name
            self.gamma = gamma
            self.alpha = tf.math.softplus(alpha) + 1e-6
            self.beta = tf.math.softplus(beta) + 1e-6

    def forward(self, z):
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            return self.alpha * tf.math.sinh(z * self.beta + self.gamma)

    def backward(self, z):
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            return (tf.math.asinh(z / self.alpha) - self.gamma) / self.beta

    def forward_derivative(self, z):
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            return self.alpha * self.beta * tf.math.cosh(z * self.beta + self.gamma)

        
