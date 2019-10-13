import tensorflow as tf

# this class contains definitions for different normalising flows
class RadialFlow:

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def project(z):
        return z + alpha * (z - beta)

    
