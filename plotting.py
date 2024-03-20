import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import seaborn as sns

from benchmarks import BENCHMARK_LIST
from dataset import BETHDataset, GaussianDataset

matplotlib.use('Agg')
sns.set()

def plot_benchmark(model):
    # fig = plt.gcf()
    palette = sns.light_palette("teal", reverse=True, as_cmap=True) # sns.color_palette("Blues", as_cmap=True)
    min_ds, max_ds = -3, 3
    xx, yy = np.meshgrid(np.linspace(min_ds, max_ds, 50), np.linspace(min_ds, max_ds, 50))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # plot the levels lines and the points
    plt.contourf(xx, yy, Z, cmap=palette)


def plot_line(x, y, filename="", xlabel="", ylabel=""):
    """
    Plotting a line (train and val loss)
    """
    fig, ax = plt.subplots()

    sns.lineplot(x=x, y=y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.savefig(os.path.join("results", f"{filename}.png"))
    plt.close()


def plot_2d_dataset(dataset_list, label_list, prefix="results/dataset", suffix="", xlim=None, ylim=None):
    """
    Plotting in 2d, largely for the embedded space
    """
    title = prefix
    fig, ax = plt.subplots()
    colors = sns.color_palette("colorblind", len(label_list))

    for i in range(len(dataset_list)):
        sns.scatterplot(x=dataset_list[i][:, 0], y=dataset_list[i][:, 1], color=colors[i], label=label_list[i])
        title += "_" + str(label_list[i])
    ax.legend()
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if suffix != "":  # Pretty ending
        suffix = "_" + suffix
    return title + suffix + ".png"


def plot_data(dataset_list, label_list, base_dataset, prefix="dataset", suffix="", xlim=None, ylim=None):
    """
    Determines which dataset plotting method to call before passing over the dataset information.

    param:
        dataset_list: list of lists and datasets from dataset.py to be plotted
        label_list: list of string labels for each passed dataset
        prefix: string to prepend to the file name
        suffix: string to append to the file name (start with "_" for pretty naming)
    """
    if len(dataset_list) != len(label_list):
        raise Exception("Unequal length of lists passed") # Dumb check

    # Check if an sklearn model is in the list
    SKLEARN_LIST = [b for b in BENCHMARK_LIST if b != "dose"]
    benchmark, benchmark_name, index = None, None, None
    if any(label in SKLEARN_LIST for label in label_list):
        index = [label in SKLEARN_LIST for label in label_list].index(True)
        benchmark_name = label_list[index]
        benchmark = dataset_list[index]
        del dataset_list[index]
        del label_list[index]

    plt.figure(figsize=(16,10))
    # Plot benchmark if present
    if benchmark != None:
        plot_benchmark(benchmark)

    # Pass to that plotting function
    if base_dataset == "":
        filename = plot_2d_dataset(dataset_list, label_list, prefix, suffix, xlim=xlim, ylim=ylim)
    else:
        filename = base_dataset.plot(dataset_list, label_list, prefix, suffix)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join("results", f"{filename}.png"))
    plt.close()
