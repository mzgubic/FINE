import tensorflow as tf
import tensorflow.contrib.layers as layers
import math as m
import numpy as np

class FlowModel:

    def __init__(self, number_warps, flow_model):
        self.number_warps = number_warps
        self.flow_model = flow_model
        self.trafos = [] # this will hold the individual flow transformations later on

        print("starting tensorflow session ...")
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()

    def std_normal(self, x):
        return 1.0 / tf.sqrt(2 * m.pi) * tf.exp(-0.5 * tf.math.square(x))

    def build_param_network(self, intensor, num_layers, num_units, num_params):
        with tf.variable_scope("flow_paramnet"):
            lay = intensor

            for cur_lay in range(num_layers):
                lay = layers.relu(lay, num_units)

            outtensor = layers.linear(lay, num_params)

        return outtensor
    
    def build_logcdf(self, x, theta, trafos):
        """
        Add the TF graph objects implementing a conditional model p(x|theta).
        Some internal random variable z is related to x through 'number_warps' applications
        of the normalising flow transformation.
        """
        eps = 1e-6
        
        # compute the inverse of the overall flow transformation and also pick up the
        # Jacobian terms on the way
        with tf.variable_scope("flow"):
            x0 = x
            jacs = []
            for cur_trafo in reversed(trafos):
                x0 = cur_trafo.backward(x0)
                jacs.append(tf.log(cur_trafo.forward_derivative(x0) + eps))
                
            logcdf = tf.math.log(self.std_normal(x0) + eps) - tf.add_n(jacs)
            
        return tf.squeeze(logcdf)

    def build_fisher(self, x, theta, trafos):
        """
        Add the TF graph objects which, given a fitted normalising flow, compute
        the Fisher information (density).
        x ... a vector of random numbers distributed according to the innermost distribution
              of the normalising flow, i.e. a standard normal distribution. These will be used
              to perform the actual Monte Carlo integration.
        """
        with tf.variable_scope("fisher"):
            self.xk = x
            self.fisher_densities = []
            for cur_trafo in trafos: # this time, need to iterate in the forward direction
                self.fisher_densities.append(tf.math.reduce_mean(tf.hessians(cur_trafo.forward(self.xk), theta)))
                #self.fisher_densities.append(tf.hessians(cur_trafo.forward(self.xk), theta))
                self.xk = cur_trafo.forward(self.xk) # propagate them to the next transformation in the flow

            return tf.add_n(self.fisher_densities) # add up all contributions to give the full Fisher information
    
    def init(self):
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
    
    def build(self):        
        with self.graph.as_default():
            self.x_in = tf.placeholder(tf.float32, [None, 1], name = 'x_in')
            self.theta_in = tf.placeholder(tf.float32, [None, 1], name = 'theta_in')
            self.rnd_in = tf.placeholder(tf.float32, [None, 1], name = 'rnd_in')

            # construct the network computing the parameters of the flow transformations
            self.flow_params = self.build_param_network(intensor = self.theta_in, num_layers = 2, num_units = self.number_warps * 2, num_params = self.number_warps * 2)
            
            # initialise the flow transformations
            self.alphas = self.flow_params[:,:self.number_warps]
            self.betas = self.flow_params[:,self.number_warps:]
            
            for cur in range(self.number_warps):
                cur_alpha = tf.expand_dims(self.alphas[:,cur], axis = 1)
                cur_beta = tf.expand_dims(self.betas[:,cur], axis = 1)
                self.trafos.append(self.flow_model(alpha = cur_alpha, beta = cur_beta, name = "flow_trafo_{}".format(cur)))

            self.logcdf = self.build_logcdf(self.x_in, self.theta_in, trafos = self.trafos)

            # add the loss for the training of the conditional density estimator
            self.loss = -tf.math.reduce_sum(self.logcdf, axis = 0) # sum along the batch dimension
            
            # add optimiser
            self.fit_step = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(self.loss)

            # add some more operations that compute the Fisher information
            self.fisher = self.build_fisher(self.rnd_in, self.theta_in, self.trafos)
            
    def evaluate(self, x, theta):
        """
        Evaluate the modelled density p(x|theta)
        """
        with self.graph.as_default():
            val = self.sess.run(self.logcdf, feed_dict = {self.x_in: x, self.theta_in: theta})
        return val

    def evaluate_fisher(self, theta, num_samples = 1000):
        """
        Compute the Fisher information w.r.t. theta.
        """
        # generate some random numbers for the MC integration
        rnd = np.expand_dims(np.random.normal(loc = 0.0, scale = 1.0, size = num_samples), axis = 1)
        with self.graph.as_default():
            fisher = self.sess.run(self.fisher, feed_dict = {self.rnd_in: rnd, self.theta_in: theta})

        return fisher

    def evaluate_fisher_with_debug(self, theta, num_samples = 5):
        rnd = np.expand_dims(np.random.normal(loc = 0.0, scale = 1.0, size = num_samples), axis = 1)

        with self.graph.as_default():
            dens = self.sess.run(self.fisher_densities[0], feed_dict = {self.rnd_in: rnd, self.theta_in: theta})
            xk = self.sess.run(self.xk, feed_dict = {self.rnd_in: rnd, self.theta_in: theta})

        print("dens = {}".format(dens))
        print("xk = {}".format(xk))
    
    def fit(self, x, theta, number_steps = 10):
        for cur_step in range(number_steps):
            with self.graph.as_default():
                self.sess.run(self.fit_step, feed_dict = {self.x_in: x, self.theta_in: theta})
                if cur_step % 100:
                    loss = self.sess.run(self.loss, feed_dict = {self.x_in: x, self.theta_in: theta})
                    print("loss = {}".format(loss))
    
    def evaluate_with_debug(self, x, theta):
        with self.graph.as_default():
            debug_flow_params = self.sess.run(self.flow_params, feed_dict = {self.x_in: x, self.theta_in: theta})
            debug_alphas = self.sess.run(self.alphas[:,3], feed_dict = {self.x_in: x, self.theta_in: theta})
            debug_betas = self.sess.run(self.betas[:,3], feed_dict = {self.x_in: x, self.theta_in: theta})
            debug_flowout = self.sess.run(self.trafos[1].backward(self.x_in), feed_dict = {self.x_in: x, self.theta_in: theta})
            debug_x = self.sess.run(self.x_in, feed_dict = {self.x_in: x, self.theta_in: theta})
        print("flow_params = {}".format(debug_flow_params))
        print("alphas = {}".format(debug_alphas))
        print("betas = {}".format(debug_betas))
        print("flowout = {}".format(debug_flowout))
        print("x_int = {}".format(debug_x))
        

