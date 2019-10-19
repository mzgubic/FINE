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
    return np.random.normal(loc = np.full_like(theta, 0.0), scale = theta * 2, size = len(theta))

def evaluate_CDE(theta, x):
    """
    Evaluate the conditional density p(x|theta)
    """
    return np.log(1.0 / np.sqrt(2 * np.pi * np.square(theta * 2)) * np.exp(-0.5 * np.square((x / (theta * 2)))))

def generate_data(nsamples, theta_low, theta_high):
    """
    Generate pairs (x, theta), where theta is drawn from a uniform distribution and x comes
    from the original conditional model.
    """
    theta = np.random.uniform(low = theta_low, high = theta_high, size = nsamples)
    x = sample_CDE(theta = theta)
    return np.expand_dims(x, axis = 1), np.expand_dims(theta, axis = 1)

def run():
    print("running with tensorflow version {}".format(tf.__version__))

    # prepare samples from the original conditional distribution that is to be estimated
    nsamples = 300
    theta_low = 2
    theta_high = 6
    data, theta = generate_data(nsamples, theta_low, theta_high)

    # create a simple scatter plot to visualise this datset
    Plotter.scatter_plot(xs = [theta], ys = [data], labels = ["data"], outfile = "data.pdf", xlabel = r'$\theta$', ylabel = r'$x$')

    # now build a model to implement the conditional density
    mod = FlowModel(number_warps = 5, flow_model = TombsFlow)
    mod.build()
    mod.init()

    mod.fit(x = data, theta = theta, number_steps = 4000)
    
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

    # make some cross sectional plots through the CDE landscape
    theta = np.expand_dims(np.linspace(theta_low, theta_high, 100), axis = 1)
    x = np.zeros_like(theta)
    crosssection = mod.evaluate(x = x, theta = theta)
    crosssection_truth = evaluate_CDE(x = x, theta = theta)
    Plotter.scatter_plot(xs = [theta, theta], ys = [crosssection, crosssection_truth], labels = [r'$p(x = 0|\theta)$', 'truth'], outfile = "x_0.pdf", xlabel = r'$\theta$')
    
    # evaluate the Fisher information
    theta = np.linspace(theta_low, theta_high, 20)
    fisher_tf = []
    fisher_analytic = []
    for cur_theta in theta:
        fisher_tf.append(mod.evaluate_fisher(theta = [[cur_theta]]))
        fisher_analytic.append(2.0 / cur_theta**2)

    print(fisher_tf)
        
    #Plotter.scatter_plot(xs = [theta, theta], ys = [fisher, fisher_analytic], labels = ["FINE", "analytic"], outfile = "fisher.pdf", xlabel = r'$\theta$', ylabel = "Fisher information")
    Plotter.scatter_plot(xs = [theta], ys = [fisher_tf], labels = ["FINE"], outfile = "fisher.pdf", xlabel = r'$\theta$', ylabel = "Fisher information")

    # x, y = mod.evaluate_fisher_alternative(theta = [[4.0]])
    # Plotter.scatter_plot(xs = [x], ys = [y], labels = [""], outfile = "testgradient.pdf")
    
    # sampledata = mod.evaluate_fisher_alternative(theta = [[4.0]])
    # Plotter.histogram(sampledata, "testsampling.pdf")
    
if __name__ == "__main__":
    parser = ArgumentParser(description = "launch training campaign")
    args = vars(parser.parse_args())

    run()

