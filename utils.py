import numpy as np

Base_DIR = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV"
SAVE_DIR = "/Users/mac/Desktop/RLPortfolio/DirichletPortfolio"

def sigmoid(x):
    x = np.clip(x, -10, 10)
    return 1. / (1. + np.exp(-x))

def exp(x):
    return np.exp(x)


if __name__ == "__main__":

    start = 0.000001
    end = 0.0025
    steps_done = 0
    p = end + (start-end) * np.exp(-steps_done/5000)

    for i in range(100000):
        p = end + (start - end) * np.exp(-steps_done / 10000)
        steps_done += 1
        print(p)


