import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import torch
from Distribution import Dirichlet

dirichlet = Dirichlet(torch.tensor([1.2935, 1.3279, 2.0000]))
sample = dirichlet.sample(sample_shape=[100000])

sample = sample[:, :2]
X = sample[:, 0].numpy()
Y = sample[:, 1].numpy()

mean = [0.2799, 0.2873]
mode = [0.1810, 0.2022]

plt.figure(figsize=(10,8))
sns.jointplot(x=X, y=Y, kind="hex")
sns.scatterplot()
plt.scatter(x=mean[0], y=mean[1], color="red", label="mean")
plt.scatter(x=mode[0], y=mode[1], color="yellow", label="mode")
plt.scatter(x=0.33, y=0.33, color="orange", label="median")

plt.legend()
plt.show()

