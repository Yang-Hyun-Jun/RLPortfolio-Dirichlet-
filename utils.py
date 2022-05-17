import numpy as np

Base_DIR = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV"
SAVE_DIR = "/Users/mac/Desktop/RLPortfolio/DirichletPortfolio"

def sigmoid(x):
    x = np.clip(x, -10, 10)
    return 1. / (1. + np.exp(-x))

def exp(x):
    return np.exp(x)


if __name__ == "__main__":
    import torch
    import torch.nn as nn

    a = torch.rand(size=(1, 4))
    print(torch.softmax(a, dim=1))