import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    common_path1 = "/Users/mac/Desktop/RLPortfolio"
    common_path2 = "/Metrics/cum_fees"

    # term o model
    profitlosses1 = pd.read_csv(common_path1 + "/DirichletPortfolio (term o)" + common_path2).iloc[:,1]
    # term x model
    profitlosses2 = pd.read_csv(common_path1 + "/DirichletPortfolio (term x)" + common_path2).iloc[:,1]


    #Visualizing
    fig, ax = plt.subplots(figsize=(60, 8), facecolor="w")
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.yaxis.tick_right()
    ax.set_facecolor("lightgray")
    ax.set_ylabel("Fee")
    ax.set_xlabel("Time step")
    plt.title("Fee")
    plt.plot(profitlosses1, label="term o")
    plt.plot(profitlosses2, label="term x")


    xticks = [int(i) for i in np.linspace(0, len(profitlosses1), 6)]
    plt.xticks(xticks)
    plt.grid(True, color="w", alpha=0.5)
    plt.legend()
    plt.show()
    fig.savefig("/Users/mac/Desktop/RLPortfolio/TestingFee")