from math import log
import os
import pickle

import numpy as np
import torch
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal
import tqdm

from dose import DoSE_SVM, kl_divergence


## scikit-learn models
def train_sklearn(epoch, dataset, model):
    train_loss = 0
    # fit the data and tag outliers
    model.fit(dataset.data)
    train_scores = model.decision_function(dataset.data) # positive distances for inlier, negative for outlier
    train_loss = -1 * np.average(train_scores) # reverse signage
    return train_loss, model


def validate_sklearn(epoch, dataset, model):
    # fit the data and tag outliers
    val_scores = model.decision_function(dataset.data) # positive distances for inlier, negative for outlier
    val_loss = -1 * np.average(val_scores) # reverse signage
    return val_loss


def test_sklearn(seed, args, train_dataset, test_dataset):
    # Load the models (5 of each)
    filename = os.path.join("results", f"{args.dataset}_{args.benchmark}_{seed}.pth")
    model = pickle.load(open(filename, 'rb'))
    # Run model on testing dataset
    y_pred = model.predict(test_dataset.data)
    if type(model) == IsolationForest or type(model) == SGDOneClassSVM:
        y_pred = [0 if y == 1 else 1 for y in y_pred] # iForest sets 1 as inlier and -1 as outlier
    else:
        y_pred = [0 if y == -1 else y for y in y_pred]
    outlier_preds = y_pred
    # Compare labels/predictions to labels
    return outlier_preds


## VAE + DoSE(SVM)
def train_vae(epoch, data_loader, model, prior, optimiser, device):
    model.train()
    zs = []
    train_loss = 0
    for i, (x, y) in enumerate(tqdm.tqdm(data_loader, leave=False)):
        x, y = x.to(device=device, non_blocking=True), y.to(device=device, non_blocking=True)
        observation, posterior, z = model(x)
        loss = -observation.log_prob(x) + kl_divergence(z, posterior, prior)
        loss = -torch.logsumexp(-loss.view(loss.size(0), -1), dim=1).mean() - log(1)
        zs.append(z.detach())  # Store posterior samples
        train_loss += loss.item()
        
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    return train_loss / len(data_loader), torch.cat(zs)
    

def validate_vae(epoch, data_loader, model, prior, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm.tqdm(data_loader, leave=False)):
            x, y = x.to(device=device, non_blocking=True), y.to(device=device, non_blocking=True)
            observation, posterior, z = model(x)
            loss = -observation.log_prob(x) + kl_divergence(z, posterior, prior)
            val_loss += -torch.logsumexp(-loss.view(loss.size(0), -1), dim=1).mean() - log(1)
    return val_loss.item() / len(data_loader)


def test_vae(seed, args, train_dataset, test_dataset):
    # Calculate result over ensemble of trained models
    # Load dataset summary statistics
    train_summary_stats = torch.load(os.path.join("stats", f"{args.dataset}_{args.benchmark}_{args.seed}_stats_train.pth"))
    val_summary_stats = torch.load(os.path.join("stats", f"{args.dataset}_{args.benchmark}_{args.seed}_stats_val.pth"))
    test_summary_stats = torch.load(os.path.join("stats", f"{args.dataset}_{args.benchmark}_{args.seed}_stats_test.pth"))
    print(f"train shape: {train_summary_stats.shape}")
    print(f"test shape: {test_summary_stats.shape}")

    dose_svm = DoSE_SVM(train_summary_stats)
    outlier_preds = dose_svm.detect_outliers(test_summary_stats)
    return outlier_preds


def get_marginal_posterior(data_loader, model, device):
    model.eval()
    posteriors = []
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm.tqdm(data_loader, leave=False)):
            x, y = x.to(device=device, non_blocking=True), y.to(device=device, non_blocking=True)
            posteriors.append(model.encode(x))
    means, stddevs = torch.cat([p.mean for p in posteriors], dim=0), torch.cat([p.stddev for p in posteriors], dim=0)
    mix = Categorical(torch.ones(means.size(0), device=device))
    comp = Independent(Normal(means, stddevs), 1)
    return MixtureSameFamily(mix, comp)
