import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    common_path1 = "/Users/mac/Desktop/RLPortfolio/DirichletPortfolio"
    common_path2 = "/Metrics/Profitloss_test"

    # Delta 0.0 model
    profitlosses1 = pd.read_csv(common_path1 + " (delta 0.0)" + common_path2).iloc[:, 1]
    # Delta 0.03 model
    profitlosses2 = pd.read_csv(common_path1 + " (delta 0.03)" + common_path2).iloc[:, 1]
    # Delta 0.05 model
    profitlosses3 = pd.read_csv(common_path1 + " (delta 0.05)" + common_path2).iloc[:, 1]
    # Delta 0.07 model (Base)
    profitlosses4 = pd.read_csv(common_path1 + " (base)" + common_path2).iloc[:, 1]
    # Delta 0.1 model
    profitlosses5 = pd.read_csv(common_path1 + " (delta 0.1)" + common_path2).iloc[:, 1]
    # Delta 0.13 model
    profitlosses6 = pd.read_csv(common_path1 + " (delta 0.13)" + common_path2).iloc[:, 1]
    # B&H
    profitlosses7 = pd.read_csv("/Users/mac/Desktop/RLPortfolio/B&H").iloc[:, 1]

    # Visualizing
    fig, ax = plt.subplots(figsize=(60, 8), facecolor="w")
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.yaxis.tick_right()
    ax.set_facecolor("lightgray")
    ax.set_ylabel("Profitloss")
    ax.set_xlabel("Time step")
    plt.title("Profitloss")
    plt.plot(profitlosses1, label="Delta 0.0")
    plt.plot(profitlosses2, label="Delta 0.03")
    plt.plot(profitlosses3, label="Delta 0.05")
    plt.plot(profitlosses4, label="Delta 0.07")
    plt.plot(profitlosses5, label="Delta 0.1")
    plt.plot(profitlosses6, label="Delta 0.13")
    plt.plot(profitlosses7, label="B&H")

    xticks = [int(i) for i in np.linspace(0, len(profitlosses1), 6)]
    plt.xticks(xticks)
    plt.grid(True, color="w", alpha=0.5)
    plt.legend()
    plt.show()
    fig.savefig("/Users/mac/Desktop/RLPortfolio/TestingDelta")
