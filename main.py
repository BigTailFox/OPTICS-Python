# coding=utf-8

# created by leoherz, 2020/04/17
# an example of OPTICS, on data set:

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from optics import OPTICS

COLORS = ['r', 'g', 'b', 'k', 'c', 'y', 'orange']

if __name__ == "__main__":
    df = pd.read_table("cluster_data_set/Aggregation.txt",
                       delim_whitespace=True)
    print(df.describe())
    fig0 = plt.figure("Aggregation: ground truth")
    for i in range(len(df)):
        plt.scatter(df.iloc[i, 0], df.iloc[i, 1],
                    color=COLORS[df.iloc[i, 2] % len(COLORS)])

    # remove label for clustering task.
    df1 = df.iloc[:, :2]
    # run optics cluster.
    model = OPTICS(df1)
    model.optics(Eps=10000, MinPts=5)
    # show ordered reachable distance and core distance.
    fig1 = plt.figure("ordered reachable and core distances")
    for i in range(model.len):
        rd = model.reachable_distances[model.result_queue[i]]
        cd = model.core_distances[model.result_queue[i]]
        category = model.category_queue[model.result_queue[i]]
        plt.bar(i, rd, color=COLORS[category % len(COLORS)])
    plt.ylim(-0.01, 2)

    # extract cluster result and visualize
    model.cluster_extract(1.5)
    fig2 = plt.figure("Aggregation: result")
    for i in range(model.len):
        category = model.category_queue[i]
        plt.scatter(df.iloc[i, 0], df.iloc[i, 1],
                    color=COLORS[category % len(COLORS)])

    plt.show()
