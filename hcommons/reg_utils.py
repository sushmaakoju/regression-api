from __future__ import absolute_import
import getopt, sys
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

__all__ = ['save_plot_as_image']

def save_plot_as_image(fig, filepath, filename):
    """
    Saves any plot as an image file. 
    :param figure: The plot to save
    :param filename: The filename to save to. The extension of the filename determines the format (jpg, png, gif, ?)
    """
    if fig != None:
        fig.savefig(filepath+"/"+filename+'.png', bbox_inches='tight', dpi=300)