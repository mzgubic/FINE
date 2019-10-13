import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata

class Plotter:

    @staticmethod
    def scatter_plot(x, y, outfile, xlabel = "", ylabel = ""):
        fig = plt.figure(figsize = (6, 5))
        ax = fig.add_subplot(111)
        ax.scatter(x, y, s = 1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.savefig(outfile)

    @staticmethod
    def heatmap(x, y, z, outfile, xlabel = "", ylabel = "", density = 1000):
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        
        fig = plt.figure(figsize = (6, 5))
        ax = fig.add_subplot(111)
        xi = np.linspace(np.min(x), np.max(x), density)
        yi = np.linspace(np.min(y), np.max(y), density)
        zi = griddata(x, y, z, xi, yi, interp = 'linear')
        ax.contourf(xi, yi, zi, 15, vmax = np.max(zi), vmin = np.min(zi))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.savefig(outfile)

