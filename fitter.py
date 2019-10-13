import tensorflow as tf

class Fitter:

    @staticmethod
    def fit(flow_model, x, theta):
        """
        Fit the 'flow_model' to model the conditional density from which the pairs (x, theta)
        were drawn.
        """

        
