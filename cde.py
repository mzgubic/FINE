from argparse import ArgumentParser
import tensorflow as tf
import numpy as np

from plotting import Plotter
from flow_model import FlowModel
from flows import RadialFlow

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
    return np.log(1.0 / np.sqrt(2 * np.pi * np.square(theta)) * np.exp(-0.5 * np.square((x / theta))))

def generate_data(nsamples):
    """
    Generate pairs (x, theta), where theta is drawn from a uniform distribution and x comes
    from the original conditional model.
    """
    theta = np.random.uniform(low = 2, high = 4, size = nsamples)
    x = sample_CDE(theta = theta)
    return np.expand_dims(x, axis = 1), np.expand_dims(theta, axis = 1)

def run():
    print("running with tensorflow version {}".format(tf.__version__))

    # prepare samples from the original conditional distribution that is to be estimated
    nsamples = 100
    data, theta = generate_data(nsamples)

    # create a simple scatter plot to visualise this datset
    Plotter.scatter_plot(x = theta, y = data, outfile = "data.pdf", xlabel = r'$\theta$', ylabel = r'$x$')

    # now build a model to implement the conditional density
    mod = FlowModel(number_warps = 3, flow_model = RadialFlow)
    mod.build()
    mod.init()
    mod.fit(x = data, theta = theta, number_steps = 1000)

    # now evaluate the fitted density model and create a heatmap
    density = 10
    x_range = np.linspace(-4, 4, density)
    theta_range = np.linspace(2, 4, density)
    evalpts = np.array(np.meshgrid(x_range, theta_range)).T.reshape(-1, 2)

    eval_x = np.expand_dims(evalpts[:,0], axis = 1)
    eval_theta = np.expand_dims(evalpts[:,1], axis = 1)
    
    vals = mod.evaluate(x = eval_x, theta = eval_theta)
    vals_truth = evaluate_CDE(x = eval_x, theta = eval_theta)
    Plotter.heatmap(x = eval_theta, y = eval_x, z = vals, outfile = "model.pdf", xlabel = r'$\theta$', ylabel = r'$x$')
    Plotter.heatmap(x = eval_theta, y = eval_x, z = vals_truth, outfile = "truth.pdf", xlabel = r'$\theta$', ylabel = r'$x$')

    # try to evaluate the Fisher information
    theta = np.linspace(2.0, 4.0, 100)
    fisher = []
    for cur_theta in theta:
        fisher.append(mod.evaluate_fisher(theta = [[cur_theta]]))

    print(fisher)
    Plotter.scatter_plot(theta, fisher, outfile = "fisher.pdf", xlabel = r'$\theta$', ylabel = "Fisher information")
    
if __name__ == "__main__":
    parser = ArgumentParser(description = "launch training campaign")
    args = vars(parser.parse_args())

    run()

