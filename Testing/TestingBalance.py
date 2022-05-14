import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    common_path1 = "/Users/mac/Desktop/RLPortfolio"
    common_path2 = "/Metrics/Balances"

    # Dirichlet mean model
    balance1 = pd.read_csv(common_path1 + "/DirichletPortfolio (mean)" + common_path2).iloc[:,1]
    # Dirichlet mode model
    balance2 = pd.read_csv(common_path1 + "/DirichletPortfolio (mode)" + common_path2).iloc[:,1]
    # DQN model
    balance3 = pd.read_csv(common_path1 + "/DQNPortfolio" + common_path2).iloc[:,1]

    #Visualizing
    fig, ax = plt.subplots(figsize=(60, 8), facecolor="w")
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.yaxis.tick_right()
    ax.set_facecolor("lightgray")
    ax.set_ylabel("Profitloss")
    ax.set_xlabel("Time step")
    plt.title("Profitloss")
    plt.plot(balance1, label="Dirichlet (mean)", color="blue")
    plt.plot(balance2, label="Dirichlet (mode)", color="orange")
    plt.plot(balance3, label="DQN", color="green")


    xticks = [int(i) for i in np.linspace(0, len(balance1), 6)]
    plt.xticks(xticks)
    plt.grid(True, color="w", alpha=0.5)
    plt.legend()
    plt.show()
    fig.savefig("/Users/mac/Desktop/RLPortfolio/TestingBalance")