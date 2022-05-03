import DataManager
import Visualizer
import utils
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Metrics import Metrics
from Environment import environment
from Agent import agent
from Network import Actor
from Network import Critic
from Network import Score

if __name__ == "__main__":
    common_path1 = "/Users/mac/Desktop/RLPortfolio/DirichletPortfolio"
    common_path2 = "/Metrics/Profitloss_test"

    # No header model
    x_profitlosses = pd.read_csv(common_path1 + " (header x)" + common_path2).iloc[:,1]
    # DNN header model
    h_profitlosses = pd.read_csv(common_path1 + " (enc X, share X, delta 0.07)" + common_path2).iloc[:,1]
    # Simple header model
    s_profitlosses = pd.read_csv(common_path1 + " (simple header)" + common_path2).iloc[:,1]

    #Visualizing
    fig, ax = plt.subplots(figsize=(60, 8), facecolor="w")
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.yaxis.tick_right()
    ax.set_facecolor("lightgray")
    ax.set_ylabel("Profitloss")
    ax.set_xlabel("Time step")
    plt.title("Profitloss")
    plt.plot(x_profitlosses, color='dodgerblue', label="No header")
    plt.plot(h_profitlosses, color="red", label="Header")
    plt.plot(s_profitlosses, color="yellow", label="Simple header")

    xticks = [int(i) for i in np.linspace(0, len(x_profitlosses), 6)]
    plt.xticks(xticks)
    plt.grid(True, color="w", alpha=0.5)
    plt.legend()
    plt.show()
    fig.savefig("/Users/mac/Desktop/RLPortfolio/TestingHeader")
