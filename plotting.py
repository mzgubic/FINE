import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata

class Plotter:

    @staticmethod
    def scatter_plot(xs, ys, labels, outfile, xlabel = "", ylabel = ""):
        fig = plt.figure(figsize = (6, 5))
        ax = fig.add_subplot(111)
        for x, y, label in zip(xs, ys, labels):
            ax.scatter(x, y, s = 1, label = label)            
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.savefig(outfile)

    @staticmethod
    def heatmap(x, y, z, outfile, xlabel = "", ylabel = "", density = 1000):
        x, y, z = x.flatten(), y.flatten(), z.flatten()

        fig = plt.figure(figsize = (6, 5))
        ax = fig.add_subplot(111)
        xi = np.linspace(np.min(x), np.max(x), density)
        yi = np.linspace(np.min(y), np.max(y), density)

        zi = griddata(x, y, z, xi, yi, interp = 'linear')
        cont = ax.contourf(xi, yi, zi, 55, vmax = np.max(zi), vmin = np.min(zi))
        fig.colorbar(cont)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.savefig(outfile)

