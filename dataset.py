from os import path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.decomposition import PCA
import torch
from torch.utils.data import TensorDataset
from torchvision import transforms

# DATASETS dictionary at bottom of file
palette_name = "bright" #"colorblind"

class BETHDataset(TensorDataset):
    """
    Data collected from BETH (honeypots) and setup for unsupervised training and testing.
    """
    def __init__(self, split='train', subsample=0):
        if split == 'train':
            data = pd.read_csv("data/labelled_training_data.csv")
        elif split == 'val':
            data = pd.read_csv("data/labelled_validation_data.csv")
        elif split == 'test':
            data = pd.read_csv("data/labelled_testing_data.csv")
        else:
            raise Exception("Error: Invalid 'split' given")
        self.name = split
        # Select columns and perform pre-processing
        labels = pd.DataFrame(data[["sus"]])
        data = pd.DataFrame(data[["processId", "parentProcessId", "userId", "mountNamespace", "eventId", "argsNum", "returnValue"]])
        data["processId"] = data["processId"].map(lambda x: 0 if x in [0, 1, 2] else 1)  # Map to OS/not OS
        data["parentProcessId"] = data["parentProcessId"].map(lambda x: 0 if x in [0, 1, 2] else 1)  # Map to OS/not OS
        data["userId"] = data["userId"].map(lambda x: 0 if x < 1000 else 1)  # Map to OS/not OS
        data["mountNamespace"] = data["mountNamespace"].map(lambda x: 0 if x == 4026531840 else 1)  # Map to mount access to mnt/ (all non-OS users) /elsewhere
        data["eventId"] = data["eventId"]  # Keep eventId values (requires knowing max value)
        data["returnValue"] = data["returnValue"].map(lambda x: 0 if x == 0 else (1 if x > 0 else 2))  # Map to success/success with value/error
        # Extract values
        self.data = torch.as_tensor(data.values, dtype=torch.int64)
        self.labels = torch.as_tensor(labels.values, dtype=torch.int64)
        # Subsample data
        if subsample > 0:
          self.data, self.labels = self.data[::subsample], self.labels[::subsample]

        super().__init__(self.data, self.labels)

    def get_input_shape(self):  # note that does not return actual shape, but is used to configure model for categorical data
        num_classes = self.data.max(dim=0)[0] + 1
        num_classes[4] = 1011  # Manually set eventId range as 0-1011 (1010 is max value)
        return num_classes
      
    def get_distribution(self):
        return 'categorical'

    def plot(self, dataset_list, label_list, prefix="dataset", suffix=""):
        """
        Plots all datasets in dataset_list of in Gaussian style, including its own data

        param:
            dataset_list: list of lists and datasets from dataset.py to be plotted
            label_list: list of string labels for each passed dataset
            prefix: string to prepend to the file name
            suffix: string to append to the file name (start with "_" for pretty naming)
        """
        # Aggregate data and labels
        X, y, title = None, [], prefix
        for dataset, label in zip(dataset_list, label_list):            
            if isinstance(dataset, self.__class__):
                X = torch.cat([X, dataset.data], dim=0) if X != None else dataset.data
                y += [label] * dataset.data.size(0)
            else:  # Torch tensor
                X = torch.cat([X, dataset], dim=0) if X != None else dataset
                y += [label] * dataset.size(0)
            title += "_" + str(label)
        if X == None:
            raise Exception("No valid data for plotting")
        X = X.numpy()

        if not hasattr(self, 'mapper'):
            self.mapper = PCA(n_components=2).fit(X)
        x1, x2 = np.hsplit(self.mapper.transform(X), 2)

        # Plot
        fig, ax = plt.subplots()
        palette = sns.color_palette(palette_name, len(dataset_list))
        sns.scatterplot(x=x1[:, 0], y=x2[:, 0], label=y, legend="full", s=5, palette=palette)
        
        if suffix != "":
            suffix = "_" + suffix
        return prefix + title + suffix


# Quick dataset to test compilation of models
class GaussianDataset(TensorDataset):
    """
    Gaussian dataset consisting of two 2d clusters in `train` and `val`; shifting one 
    cluster within `test` and holding the other constant
    
    Binary labels for post-processing verification 
        0: data generated from same distribution as the training data 
        1: anomaly, generated from drifted/different distribution
    """
    def __init__(self, split='train', subsample=0):
        if split == 'train':
            self.data = torch.cat([0.2 * torch.randn(1000, 2) + torch.tensor([[1., 1.]]), 
                                   0.2 * torch.randn(1000, 2) + torch.tensor([[-1., -1.]])])
            self.labels = torch.tensor( [0] * len(self.data) )
        elif split == 'val':
            self.data = torch.cat([0.2 * torch.randn(100, 2) + torch.tensor([[1., 1.]]), 
                                   0.2 * torch.randn(100, 2) + torch.tensor([[-1., -1.]])])
            self.labels = torch.tensor( [0] * len(self.data) )
        elif split == 'test':
            anomalies = 0.2 * torch.randn(100, 2) + torch.tensor([[1., -1.]])
            self.data = torch.cat([0.2 * torch.randn(100, 2) + torch.tensor([[1., 1.]]), anomalies])
            self.labels = torch.cat((torch.tensor( [0] * (len(self.data) - len(anomalies)) ), torch.tensor( [1] * len(anomalies) )))
        else:
            raise Exception("Error: Invalid 'split' given")
        self.name = split
        # Subsample data
        if subsample > 0:
          self.data, self.labels = self.data[::subsample], self.labels[::subsample]
        super().__init__(self.data, self.labels)

    def get_input_shape(self):
        return (2, )
      
    def get_distribution(self):
        return 'gaussian'

    def plot(self, dataset_list, label_list, prefix="dataset", suffix=""):
        """
        Plots all datasets in dataset_list of in Gaussian style, including its own data

        param:
            dataset_list: list of lists and datasets from dataset.py to be plotted (assumed contains an GaussianDataset)
            label_list: list of string labels for each passed dataset
            prefix: string to prepend to the file name
            suffix: string to append to the file name (start with "_" for pretty naming)
        """
        title = prefix
        colors = sns.color_palette(palette_name, len(dataset_list))

        for i in range(len(dataset_list)):
            if type(dataset_list[i]) == GaussianDataset:
                plt.scatter(x=dataset_list[i].tensors[0][:, 0], y=dataset_list[i].tensors[0][:, 1], color=colors[i], label=label_list[i], edgecolors=None, linewidths=1)
            else:
                plt.scatter(x=dataset_list[i][:, 0], y=dataset_list[i][:, 1], color=colors[i], label=label_list[i], edgecolors=None, linewidths=1)
            title += "_" + str(label_list[i])
        if suffix != "": # pretty ending
            suffix = "_" + suffix
        return title + suffix


DATASETS = dict(gaussian=GaussianDataset, beth=BETHDataset)
