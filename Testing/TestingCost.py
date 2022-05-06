import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    common_path1 = "/Users/mac/Desktop/RLPortfolio/DirichletPortfolio"
    common_path2 = "/Metrics/Profitloss_test"

    # base model
    profitlosses1 = pd.read_csv(common_path1 + " (base)" + common_path2).iloc[:,1]
    # penalty 0.0 model
    profitlosses2 = pd.read_csv(common_path1 + " (penalty reward 0.0)" + common_path2).iloc[:,1]
    # penalty 0.05 model
    profitlosses3 = pd.read_csv(common_path1 + " (penalty reward 0.05)" + common_path2).iloc[:,1]
    # penalty 0.1 model
    profitlosses4 = pd.read_csv(common_path1 + " (penalty reward 0.1)" + common_path2).iloc[:,1]
    # penalty 0.15 model
    profitlosses5 = pd.read_csv(common_path1 + " (penalty reward 0.15)" + common_path2).iloc[:,1]
    # penalty 0.2 model
    profitlosses6 = pd.read_csv(common_path1 + " (penalty reward 0.2)" + common_path2).iloc[:,1]
    # B&H
    profitlosses7 = pd.read_csv("/Users/mac/Desktop/RLPortfolio/B&H").iloc[:,1]

    #Visualizing
    fig, ax = plt.subplots(figsize=(60, 8), facecolor="w")
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.yaxis.tick_right()
    ax.set_facecolor("lightgray")
    ax.set_ylabel("Profitloss")
    ax.set_xlabel("Time step")
    plt.title("Profitloss")
    plt.plot(profitlosses1, label="Base", color="red")
    plt.plot(profitlosses2, label="penalty 0.0", color="blue")
    plt.plot(profitlosses3, label="penalty 0.05", color="brown")
    plt.plot(profitlosses4, label="penalty 0.1", color="orange")
    plt.plot(profitlosses5, label="penalty 0.15", color="purple")
    # plt.plot(profitlosses6, label="penalty 0.2", color="purple")
    plt.plot(profitlosses7, label="B&H", color="green")

    xticks = [int(i) for i in np.linspace(0, len(profitlosses1), 6)]
    plt.xticks(xticks)
    plt.grid(True, color="w", alpha=0.5)
    plt.legend()
    plt.show()
    fig.savefig("/Users/mac/Desktop/RLPortfolio/TestingCost")