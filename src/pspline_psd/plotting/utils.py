import matplotlib.pyplot as plt
import os

DIR = os.path.dirname(os.path.abspath(__file__))




# def set_style():
#     """Set the default style for plotting"""
#     plt.style.use(f"{DIR}/style.mplstyle")

def set_plotting_style():
    plt.style.use("default")
    plt.rcParams["savefig.dpi"] = 100
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["font.size"] = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Liberation Sans"]
    plt.rcParams["font.cursive"] = ["Liberation Sans"]
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["image.cmap"] = "inferno"
