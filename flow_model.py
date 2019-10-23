import tensorflow as tf
import tensorflow.layers as layers
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

    def build_param_network(self, intensor, num_units, activations, num_params):
        with tf.variable_scope("flow_paramnet"):
            #intensor = tf.tile(intensor, [1, 10])
            lay = intensor

            for cur_lay, (num_units, activation) in enumerate(zip(num_units, activations)):
                lay = layers.dense(lay, num_units, activation = activation)

            outtensor = layers.dense(lay, num_params, activation = None) # a linear layer at the end

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
        x0 = x
        jacs = []
        for cur_trafo in reversed(trafos):
            x0 = cur_trafo.backward(x0)
            jacs.append(tf.log(cur_trafo.forward_derivative(x0) + eps))
                
        logcdf = tf.math.log(self.std_normal(x0) + eps) - tf.add_n(jacs)
            
        return tf.squeeze(logcdf)

    def build_cdf_sampler(self, x, trafos):
        # first, need to convert 'x' to correspond to samples taken from the actual p(x|theta)
        # just apply the flow transformation to achieve that
        x_transf = x
        for cur_trafo in trafos:
            x_transf = cur_trafo.forward(x_transf)

        return x_transf
    
    def build_fisher(self, rnd_transformed, theta, trafos):
        """
        Compute the Fisher information instead using the alternative formulation, where the second
        derivative gets replaced by the square of the first derivative. Expect this to be numerically more stable.
        """
        eps = 1e-6

        self.logcdf_fisher = self.build_logcdf(rnd_transformed, self.theta_in, trafos = self.trafos)
        
        # just need to evaluate the derivative of logcdf at these locations and take the average
        return tf.reduce_mean(tf.square(tf.gradients(self.logcdf_fisher, theta)))
    
    def init(self):
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
    
    def build(self):        
        with self.graph.as_default():
            self.x_in = tf.placeholder(tf.float32, [None, 1], name = 'x_in')
            self.theta_in = tf.placeholder(tf.float32, [None, 1], name = 'theta_in')
            self.rnd_in = tf.placeholder(tf.float32, [None, 1], name = 'rnd_in')

            # construct the network computing the parameters of the flow transformations
            self.flow_params = self.build_param_network(intensor = self.theta_in,  num_units = [30, 30, 30, 30], activations = [tf.nn.relu, tf.nn.relu, tf.math.tanh, tf.math.tanh],
                                                        num_params = self.number_warps * 3)
            
            # initialise the flow transformations
            self.alphas = self.flow_params[:, :self.number_warps]
            self.betas = self.flow_params[:, self.number_warps:2*self.number_warps]
            self.gammas = self.flow_params[:, 2*self.number_warps:3*self.number_warps]
            
            for cur in range(self.number_warps):
                cur_alpha = tf.expand_dims(self.alphas[:,cur], axis = 1)
                cur_beta = tf.expand_dims(self.betas[:,cur], axis = 1)
                cur_gamma = tf.expand_dims(self.gammas[:,cur], axis = 1)
                self.trafos.append(self.flow_model(alpha = cur_alpha, beta = cur_beta, gamma = cur_gamma, name = "flow_trafo_{}".format(cur)))

            self.logcdf = self.build_logcdf(self.x_in, self.theta_in, trafos = self.trafos)

            self.shannon_reg = -tf.math.reduce_sum(tf.math.exp(self.logcdf) * self.logcdf)
            
            # add the loss for the training of the conditional density estimator
            self.nll = -tf.reduce_mean(self.logcdf, axis = 0)
            self.loss = self.nll - 0 * self.shannon_reg
            
            # add optimiser
            self.fit_step = tf.train.AdamOptimizer(learning_rate = 0.001,
                                                   beta1 = 0.9,
                                                   beta2 = 0.999,
                                                   epsilon = 1e-08).minimize(self.loss)

            # add some more operations that compute the Fisher information
            self.sampler = self.build_cdf_sampler(self.rnd_in, trafos = self.trafos)
            #self.fisher = self.build_fisher(self.sampler, self.theta_in, self.trafos)
            self.fisher = self.build_fisher(self.rnd_in, self.theta_in, self.trafos)
            
    def evaluate(self, x, theta):
        """
        Evaluate the modelled density p(x|theta)
        """
        with self.graph.as_default():
            val = self.sess.run(self.logcdf, feed_dict = {self.x_in: x, self.theta_in: theta})
        return val
    
    def evaluate_fisher(self, theta, num_samples = 500000):
        rnd = np.expand_dims(np.random.normal(loc = 0.0, scale = 1.0, size = num_samples), axis = 1)
        theta_prepared = np.full_like(rnd, theta)
        
        with self.graph.as_default():
            rnd_transformed = self.sess.run(self.sampler, feed_dict = {self.rnd_in: rnd, self.theta_in: theta_prepared})
            fisher = self.sess.run(self.fisher, feed_dict = {self.rnd_in: rnd_transformed, self.theta_in: theta_prepared})

        return fisher
        
    def fit(self, x, theta, number_steps = 4000, batch_size = 1000):
        for cur_step in range(number_steps):

            inds = np.random.choice(len(x), batch_size)
            x_batch = x[inds]
            theta_batch = theta[inds]
            
            with self.graph.as_default():
                self.sess.run(self.fit_step, feed_dict = {self.x_in: x_batch, self.theta_in: theta_batch})
                if cur_step % 10000:
                    nll = self.sess.run(self.nll, feed_dict = {self.x_in: x_batch, self.theta_in: theta_batch})
                    reg = self.sess.run(self.shannon_reg, feed_dict = {self.x_in: x_batch, self.theta_in: theta_batch})
                    print("step {}: -log L = {:.2f} (reg = {:.2f})".format(cur_step, nll, reg))
                    
    def evaluate_gradients_with_debug(self, x, theta):
        with self.graph.as_default():
            debug_grad_alpha = self.sess.run(tf.gradients(self.loss, [self.trafos[0].alpha]), feed_dict = {self.x_in: x, self.theta_in: theta})
            debug_grad_beta = self.sess.run(tf.gradients(self.loss, [self.trafos[0].beta]), feed_dict = {self.x_in: x, self.theta_in: theta})
            debug_grad_gamma = self.sess.run(tf.gradients(self.loss, [self.trafos[0].gamma]), feed_dict = {self.x_in: x, self.theta_in: theta})
            print("grad_alpha = {}".format(debug_grad_alpha))
            print("grad_beta = {}".format(debug_grad_beta))
            print("grad_gamma = {}".format(debug_grad_gamma))
                    
    def evaluate_with_debug(self, x, theta):
        with self.graph.as_default():

            debug_alpha = self.sess.run(self.trafos[0].alpha, feed_dict = {self.x_in: x, self.theta_in: theta})
            debug_beta = self.sess.run(self.trafos[0].beta, feed_dict = {self.x_in: x, self.theta_in: theta})
            debug_gamma = self.sess.run(self.trafos[0].gamma, feed_dict = {self.x_in: x, self.theta_in: theta})
            
        print("alpha = {}".format(debug_alpha))
        print("beta = {}".format(debug_beta))
        print("gamma = {}".format(debug_gamma))        

