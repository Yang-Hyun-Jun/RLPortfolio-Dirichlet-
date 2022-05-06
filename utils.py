import numpy as np

Base_DIR = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV"
SAVE_DIR = "/Users/mac/Desktop/RLPortfolio/DirichletPortfolio"

def sigmoid(x):
    x = np.clip(x, -10, 10)
    return 1. / (1. + np.exp(-x))

def exp(x):
    return np.exp(x)

if __name__ == "__main__":
    a = np.array([1, 2, 3])
    b = np.array([3, 2, 1])
    c = np.linalg.norm(a-b)
    print(c**2)
