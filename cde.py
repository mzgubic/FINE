from argparse import ArgumentParser
import tensorflow as tf
import numpy as np

from plotting import Plotter
from flow_model import FlowModel
from flows import RadialFlow, LinearRadialFlow, TombsFlow

def sample_CDE(theta):
    """
    Draw a sample from the original conditional model, i.e. return x ~ p(x|theta).
    Theta can also be a vector.
    """
    return np.random.normal(loc = np.full_like(theta, 0.0), scale = theta, size = len(theta))

def evaluate_CDE(theta, x):
    """
    Evaluate the conditional density p(x|theta)
    """
    return np.log(1.0 / np.sqrt(2 * np.pi * np.square(theta)) * np.exp(-0.5 * np.square((x / (theta)))))

def generate_data(nsamples, theta_low, theta_high):
    """
    Generate pairs (x, theta), where theta is drawn from a uniform distribution and x comes
    from the original conditional model.
    """
    theta = np.random.uniform(low = theta_low, high = theta_high, size = nsamples)
    x = sample_CDE(theta = theta)
    return np.expand_dims(x, axis = 1), np.expand_dims(theta, axis = 1)

def make_cross_section_plot(model, theta_low, theta_high, x, outfile, density = 100):
    # make some cross sectional plots through the CDE landscape
    theta = np.expand_dims(np.linspace(theta_low, theta_high, density), axis = 1)
    x = np.ones_like(theta) * x
    crosssection = model.evaluate(x = x, theta = theta)
    crosssection_truth = evaluate_CDE(x = x, theta = theta)
    Plotter.scatter_plot(xs = [theta, theta], ys = [crosssection, crosssection_truth], labels = [r'$p(x = {}|\theta)$'.format(x[0].flatten()), 'truth'], outfile = outfile, xlabel = r'$\theta$')

def run():
    print("running with tensorflow version {}".format(tf.__version__))

    # prepare samples from the original conditional distribution that is to be estimated
    nsamples = 50000
    theta_low = 6
    theta_high = 7
    data, theta = generate_data(nsamples, theta_low, theta_high)

    # create a simple scatter plot to visualise this datset
    Plotter.scatter_plot(xs = [theta], ys = [data], labels = ["data"], outfile = "data.pdf", xlabel = r'$\theta$', ylabel = r'$x$')

    # now build a model to implement the conditional density
    mod = FlowModel(number_warps = 2, flow_model = RadialFlow)
    mod.build()
    mod.init()

    mod.fit(x = data, theta = theta, number_steps = 2000)
    
    # now evaluate the fitted density model and create a heatmap
    density = 50
    x_range = np.linspace(-4, 4, density)
    theta_range = np.linspace(theta_low, theta_high, density)
    evalpts = np.array(np.meshgrid(x_range, theta_range)).T.reshape(-1, 2)

    eval_x = np.expand_dims(evalpts[:,0], axis = 1)
    eval_theta = np.expand_dims(evalpts[:,1], axis = 1)

    vals = mod.evaluate(x = eval_x, theta = eval_theta)
    vals_truth = evaluate_CDE(x = eval_x, theta = eval_theta)
    
    Plotter.heatmap(x = eval_theta, y = eval_x, z = vals, outfile = "model.pdf", xlabel = r'$\theta$', ylabel = r'$x$')
    Plotter.heatmap(x = eval_theta, y = eval_x, z = vals_truth, outfile = "truth.pdf", xlabel = r'$\theta$', ylabel = r'$x$')

    make_cross_section_plot(mod, theta_low, theta_high, x = -6.0, outfile = "x__6.pdf")
    make_cross_section_plot(mod, theta_low, theta_high, x = -2.0, outfile = "x__2.pdf")
    make_cross_section_plot(mod, theta_low, theta_high, x = -1.0, outfile = "x__1.pdf")
    make_cross_section_plot(mod, theta_low, theta_high, x = 0, outfile = "x_0.pdf")
    make_cross_section_plot(mod, theta_low, theta_high, x = 1.0, outfile = "x_1.pdf")
    make_cross_section_plot(mod, theta_low, theta_high, x = 2.0, outfile = "x_2.pdf")
    make_cross_section_plot(mod, theta_low, theta_high, x = 6.0, outfile = "x_6.pdf")
    
    # evaluate the Fisher information
    theta = np.linspace(theta_low, theta_high, 20)
    fisher_tf = []
    fisher_analytic = []
    for cur_theta in theta:
        fisher_tf.append(mod.evaluate_fisher(theta = [[cur_theta]]))
        fisher_analytic.append(2.0 / cur_theta**2)

    print(fisher_tf)
        
    Plotter.scatter_plot(xs = [theta, theta], ys = [fisher_tf, fisher_analytic], labels = ["FINE", "analytic"], outfile = "fisher.pdf", xlabel = r'$\theta$', ylabel = "Fisher information")
    
if __name__ == "__main__":
    parser = ArgumentParser(description = "launch training campaign")
    args = vars(parser.parse_args())

    run()

