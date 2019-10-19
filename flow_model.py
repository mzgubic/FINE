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

    def build_param_network(self, intensor, num_units, num_params):
        with tf.variable_scope("flow_paramnet"):
            lay = intensor

            for cur_lay, num_units in enumerate(num_units):
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
        eps = 1e-6
        
        self.xk = x
        self.fisher_densities = []
        for cur_trafo in trafos: # this time, need to iterate in the forward direction
            #self.debug_hessian = tf.log(cur_trafo.forward_derivative(self.xk) + eps)
            self.fisher_densities.append(-tf.hessians(tf.log(cur_trafo.forward_derivative(self.xk) + eps), theta)[0])
            self.xk = cur_trafo.forward(self.xk) # propagate them to the next transformation in the flow

        return tf.add_n(self.fisher_densities) # add up all contributions to give the full Fisher information

    def build_cdf_sampler(self, x, trafos):
        # first, need to convert 'x' to correspond to samples taken from the actual p(x|theta)
        # just apply the flow transformation to achieve that
        x_transf = x
        for cur_trafo in trafos:
            x_transf = cur_trafo.forward(x_transf)

        return x_transf
    
    def build_fisher_alternative(self, x, theta, trafos):
        """
        Compute the Fisher information instead using the alternative formulation, where the second
        derivative gets replaced by the square of the first derivative. Expect this to be numerically more stable.
        """
        eps = 1e-6
        
        # just need to evaluate the derivative of logcdf at these locations and take the average
        return tf.square(tf.gradients(self.logcdf, theta))
    
    def init(self):
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
    
    def build(self):        
        with self.graph.as_default():
            self.x_in = tf.placeholder(tf.float32, [None, 1], name = 'x_in')
            self.theta_in = tf.placeholder(tf.float32, [None, 1], name = 'theta_in')
            self.rnd_in = tf.placeholder(tf.float32, [None, 1], name = 'rnd_in')

            # construct the network computing the parameters of the flow transformations
            self.flow_params = self.build_param_network(intensor = self.theta_in,  num_units = [10, 30, 40, 20], num_params = self.number_warps * 3)
            
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

            self.sampler = self.build_cdf_sampler(self.rnd_in, trafos = self.trafos)
            
            # add the loss for the training of the conditional density estimator
            self.loss = -tf.math.reduce_sum(self.logcdf, axis = 0) # sum along the batch dimension
            
            # add optimiser
            self.fit_step = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(self.loss)

            # add some more operations that compute the Fisher information
            self.fisher = self.build_fisher_alternative(self.x_in, self.theta_in, self.trafos)
            #self.fisher = self.build_fisher(self.rnd_in, self.theta_in, self.trafos)
            
    def evaluate(self, x, theta):
        """
        Evaluate the modelled density p(x|theta)
        """
        with self.graph.as_default():
            val = self.sess.run(self.logcdf, feed_dict = {self.x_in: x, self.theta_in: theta})
        return val

    def evaluate_fisher_alternative_np(self, theta, num_samples = 10000):
        rnd = np.expand_dims(np.random.normal(loc = 0.0, scale = 1.0, size = num_samples), axis = 1)
        theta_prepared = np.full_like(rnd, theta)
        
        fisher = []
        with self.graph.as_default():
            rnd_transf = self.sess.run(self.sampler, feed_dict = {self.rnd_in: rnd, self.theta_in: theta_prepared})
            for cur in rnd_transf:
                cur_fisher = self.sess.run(self.fisher, feed_dict = {self.x_in: [cur], self.theta_in: theta})
                fisher.append(cur_fisher)

        return np.mean(fisher)
    
    def evaluate_fisher_alternative(self, theta, num_samples = 50000):
        rnd = np.expand_dims(np.random.normal(loc = 0.0, scale = 1.0, size = num_samples), axis = 1)
        theta_prepared = np.full_like(rnd, theta)

        with self.graph.as_default():
            rnd_transf = self.sess.run(self.sampler, feed_dict = {self.rnd_in: rnd, self.theta_in: theta_prepared})
            fisher = self.sess.run(self.fisher, feed_dict = {self.x_in: rnd_transf, self.theta_in: theta_prepared})

        return np.mean(fisher)
    
    def evaluate_fisher(self, theta, num_samples = 1000):
        """
        Compute the Fisher information w.r.t. theta.
        """
        # generate some random numbers for the MC integration
        rnd = np.expand_dims(np.random.normal(loc = 0.0, scale = 1.0, size = num_samples), axis = 1)
        with self.graph.as_default():
            #fisher = self.sess.run(self.debug_hessian, feed_dict = {self.x_in: rnd, self.rnd_in: rnd, self.theta_in: theta})
            #print(fisher)
            fisher = self.sess.run(self.fisher / num_samples, feed_dict = {self.rnd_in: rnd, self.theta_in: theta})

        return fisher

    def evaluate_fisher_with_debug(self, theta, num_samples = 5):
        rnd = np.expand_dims(np.random.normal(loc = 0.0, scale = 1.0, size = num_samples), axis = 1)

        with self.graph.as_default():
            dens = self.sess.run(self.fisher_densities[0], feed_dict = {self.rnd_in: rnd, self.theta_in: theta})
            xk = self.sess.run(self.xk, feed_dict = {self.rnd_in: rnd, self.theta_in: theta})

        print("dens = {}".format(dens))
        print("xk = {}".format(xk))
    
    def fit(self, x, theta, number_steps = 10, burn_in_steps = 4000):
        loss_prev_avg = 1e6
        loss_cur_avg = 0.0
        cur_step = 0
        for cur_step in range(burn_in_steps):
            with self.graph.as_default():
                self.sess.run(self.fit_step, feed_dict = {self.x_in: x, self.theta_in: theta})
                loss = self.sess.run(self.loss, feed_dict = {self.x_in: x, self.theta_in: theta})
                cur_step += 1
                if cur_step % 10000:
                    print("step {}: -log L = {}".format(cur_step, loss))
            
        # while True:
        #     with self.graph.as_default():
        #         self.sess.run(self.fit_step, feed_dict = {self.x_in: x, self.theta_in: theta})
        #         loss = self.sess.run(self.loss, feed_dict = {self.x_in: x, self.theta_in: theta})
        #         loss_cur_avg += loss
        #         cur_step += 1
        #         if cur_step % 10000:
        #             print("step {}: -log L = {}".format(cur_step, loss))
        #             if loss_cur_avg < loss_prev_avg:
        #                 loss_prev_avg = loss_cur_avg
        #                 loss_cur_avg = 0
        #             else:
        #                 break

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

            # debug_logcdf = self.sess.run(self.logcdf, feed_dict = {self.x_in: x, self.theta_in: theta})
            # debug_loss = self.sess.run(self.loss, feed_dict = {self.x_in: x, self.theta_in: theta})
            # debug_A1 = self.sess.run(self.trafos[0].selected_A1, feed_dict = {self.x_in: x, self.theta_in: theta})
            # debug_A2 = self.sess.run(self.trafos[0].selected_A2, feed_dict = {self.x_in: x, self.theta_in: theta})
            # debug_B1 = self.sess.run(self.trafos[0].selected_B1, feed_dict = {self.x_in: x, self.theta_in: theta})
            # debug_B2 = self.sess.run(self.trafos[0].selected_B2, feed_dict = {self.x_in: x, self.theta_in: theta})
            # debug_z_in = self.sess.run(self.trafos[0].z_in, feed_dict = {self.x_in: x, self.theta_in: theta})
            
        print("alpha = {}".format(debug_alpha))
        print("beta = {}".format(debug_beta))
        print("gamma = {}".format(debug_gamma))
        # print("logcdf = {}".format(debug_logcdf))
        # print("loss = {}".format(debug_loss))
        # print("z_in = {}".format(debug_z_in))
        # print("A1 = {}".format(debug_A1))
        # print("A2 = {}".format(debug_A2))
        # print("B1 = {}".format(debug_B1))
        # print("B2 = {}".format(debug_B2))
        

