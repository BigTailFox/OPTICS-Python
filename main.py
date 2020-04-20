# coding=utf-8

# created by leoherz, 2020/04/17
# an example of OPTICS, on data set:

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from optics import OPTICS

COLORS = ['gold', 'darkorange', 'lightcoral',
          'seagreen', 'c', 'steelblue', 'slateblue']

if __name__ == "__main__":
    df = pd.read_table("cluster_data_set/Aggregation.txt",
                       delim_whitespace=True)
    print(df.describe())
    fig0 = plt.figure("ground truth")
    for i in range(len(df)):
        plt.scatter(df.iloc[i, 0], df.iloc[i, 1],
                    color=COLORS[df.iloc[i, 2] % len(COLORS)])

    # remove label for clustering task.
    df1 = df.iloc[:, :2]
    # run optics cluster.
    model = OPTICS(df1)
    model.optics(Eps=10000, MinPts=8)
    # show ordered reachable distance and core distance.
    fig1 = plt.figure("ordered rd with ground truth label")
    index = np.linspace(0, model.len, model.len)
    rd = []
    cd = []
    category = []
    for i in range(model.len):
        rd.append(model.reachable_distances[model.result_queue[i]])
        cd.append(model.core_distances[model.result_queue[i]])
        category.append(
            COLORS[df.iloc[model.result_queue[i], 2] % len(COLORS)])
    plt.bar(index, rd, color=category)
    plt.plot(index, rd, color='black', linewidth=0.5)
    plt.ylim(0, 5)
    plt.xlim(-5, model.len+5)
    fig2 = plt.figure("ordered rd and cd")
    plt.fill_between(index, 0, cd, alpha=0.3, color='lightcoral')
    plt.fill_between(index, 0, rd, alpha=0.3, color='slateblue')
    plt.xlim(-5, model.len+5)
    plt.ylim(0, 5)

    # extract cluster result and visualize
    model.cluster_extract(1.65)
    fig3 = plt.figure("clustering result")
    cluster = []
    for i in range(model.len):
        c = model.category_queue[model.result_queue[i]]
        if c == -1:
            cluster.append("black")
        else:
            cluster.append(
                COLORS[c % len(COLORS)])
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], color=cluster)

    plt.show()
