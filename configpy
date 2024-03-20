import argparse

import numpy as np
import torch

from benchmarks import BENCHMARK_LIST
from dataset import DATASETS

def configure():
    parser = argparse.ArgumentParser(description='Michelin Star Restaurant Process')
    # General flags
    parser.add_argument('--dataset', type=str, default="beth", choices=list(DATASETS.keys()), metavar='D', help='Dataset selection')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
    parser.add_argument('--subsample', type=int, default=0, metavar='S', help='Factor by which to subsample the dataset')
    parser.add_argument('--vis-latents', action='store_true', help='True if want to visualise latent space')
    parser.add_argument('--vis', action='store_true', help='True if want to visualise dataset (and each epoch)')
    # Training/Testing flags
    parser.add_argument('--test', action='store_true', help='Test benchmarks')
    parser.add_argument('--train', action='store_true', help='Train benchmarks')
    parser.add_argument('--batch-size', type=int, default=128, metavar='B', help='Minibatch size')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='Training epochs')
    parser.add_argument('--patience', type=int, default=3, metavar='P', help='Early stopping patience')
    # Model flags
    parser.add_argument('--benchmark', type=str, default="rcov", choices=BENCHMARK_LIST, help='Override fitting of VAE model with specified benchmark')
    parser.add_argument('--outliers-fraction', type=float, default=0.05, help='Assumed proportion of the data that is an outlier') # used in rcov and ifor
    # # VAE
    parser.add_argument('--latent-size', type=int, default=2, metavar='Z', help='Latent size')
    parser.add_argument('--hidden-size', type=int, default=64, metavar='H', help='Hidden size')
    parser.add_argument('--learning-rate', type=float, default=0.003, metavar='L', help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.1, metavar='W', help='Weight decay')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available() and not args.disable_cuda
    args.device = torch.device('cuda' if use_cuda else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    return args
