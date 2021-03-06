import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    common_path1 = "/Users/mac/Desktop/RLPortfolio/DirichletPortfolio"
    common_path2 = "/Metrics/Profitloss_test"

    # Off policy model
    profitlosses1 = pd.read_csv(common_path1 + " (base)" + common_path2).iloc[:,1]
    # On policy model
    profitlosses2 = pd.read_csv(common_path1 + " (On policy)" + common_path2).iloc[:,1]
    # B&H
    profitlosses3 = pd.read_csv("/Users/mac/Desktop/RLPortfolio/B&H").iloc[:,1]

    #Visualizing
    fig, ax = plt.subplots(figsize=(60, 8), facecolor="w")
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.yaxis.tick_right()
    ax.set_facecolor("lightgray")
    ax.set_ylabel("Profitloss")
    ax.set_xlabel("Time step")
    plt.title("Profitloss")
    plt.plot(profitlosses1, label="Off policy", color="red")
    plt.plot(profitlosses2, label="On policy", color="blue")
    plt.plot(profitlosses3, label="B&H", color="green")

    xticks = [int(i) for i in np.linspace(0, len(profitlosses1), 6)]
    plt.xticks(xticks)
    plt.grid(True, color="w", alpha=0.5)
    plt.legend()
    plt.show()
    fig.savefig("/Users/mac/Desktop/RLPortfolio/TestingONOFF")