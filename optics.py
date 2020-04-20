# coding=utf-8

# created by leoherz, 2020/04/17
# classic data mining algorithm OPTICS-DBSCAN implemented with python

import numpy as np
import pandas as pd


class OPTICS(object):

    def __init__(self, dataframe: pd.DataFrame):
        self.data = dataframe
        self.len = len(dataframe)
        self.ordered_queue: list = []
        self.result_queue: list = []
        self.cluster_labels = np.zeros(self.len, dtype=int)
        self.visited = np.zeros(self.len, dtype=int)
        self.adjacency_matrix = np.full((self.len, self.len), -1, dtype=float)
        # the index that can sort adjacency_matrix by rows.
        self.sort_matrix = np.zeros((self.len, self.len), dtype=int)
        # where 0 means not core point and >0 means core point.
        self.core_points = np.zeros(self.len, dtype=int)
        self.core_distances = np.full(self.len, -1, dtype=float)
        self.reachable_distances = np.full(self.len, -1, dtype=float)

    def __initialize(self, Eps, MinPts):
        # create full connected graph of data entries.
        for i in range(self.len):
            p1 = self.data.iloc[i, :]
            for j in range(self.len):
                p2 = self.data.iloc[j, :]
                if self.adjacency_matrix[j][i] != -1:
                    self.adjacency_matrix[i][j] = self.adjacency_matrix[j][i]
                else:
                    self.adjacency_matrix[i][j] = OPTICS.distance(p1, p2)
        # create sorting matrix that sorts distances in adjacency_matrix by rows, ascending order.
        for i in range(self.len):
            self.sort_matrix[i, :] = np.argsort(self.adjacency_matrix[i, :])
        # find all core points and how many neighbor points belonging to it,
        # 0 means it's not a core point, 5 means it has 5 neighbors(NOT including itself) within eps.
        self.__find_core_point(Eps, MinPts)
        # initialize other variables.

    def clear(self):
        self.ordered_queue = []
        self.result_queue = []
        self.cluster_labels = np.zeros(self.len, dtype=int)
        self.visited = np.zeros(self.len, dtype=int)
        self.adjacency_matrix = np.full((self.len, self.len), -1, dtype=float)
        # the index that can sort adjacency_matrix by rows.
        self.sort_matrix = np.zeros((self.len, self.len), dtype=int)
        # where 0 means not core point and >0 means core point.
        self.core_points = np.zeros(self.len, dtype=int)
        self.core_distances = np.full(self.len, -1, dtype=float)

    @staticmethod
    def distance(p1: pd.Series, p2: pd.Series):
        return (sum((p1 - p2) ** 2)) ** 0.5

    def __find_core_point(self, Eps, MinPts):
        for i in range(self.len):
            # MinPts-th clear distance.
            d = self.adjacency_matrix[i, self.sort_matrix[i, MinPts]]
            if d <= Eps:
                self.core_distances[i] = d
                self.core_points[i] = MinPts
                # find all neighbors within Eps.
                for j in range(MinPts + 1, self.len):
                    d_new = self.adjacency_matrix[i, self.sort_matrix[i, j]]
                    if d_new <= Eps:
                        self.core_points[i] = j
                    else:
                        break

    def __insert_ordered_queue(self, cp):
        for j in range(1, self.core_points[cp] + 1):
            index = self.sort_matrix[cp, j]
            # skip visited points.
            if self.visited[index] == 1:
                continue
            # calculate reachable distance.
            rd = max(self.core_distances[cp], self.adjacency_matrix[cp, index])
            # if point not in ordered queue, insert it and re-order.
            if index not in self.ordered_queue:
                self.reachable_distances[index] = rd
                self.ordered_queue.append(index)
                self.ordered_queue.sort(
                    key=lambda x: self.reachable_distances[x])
            # if point in ordered queue
            else:
                # if new reachable distance is smaller, update rd list, and re-order.
                if rd < self.reachable_distances[index]:
                    self.reachable_distances[index] = rd
                    self.ordered_queue.sort(
                        key=lambda x: self.reachable_distances[x])

    def optics(self, Eps, MinPts):
        self.__initialize(Eps, MinPts)
        for i in range(self.len):
            # from data set choose an unvisited core point, add to result queue.
            if self.visited[i] == 0 and self.core_points[i] != 0:
                self.visited[i] = 1
                self.result_queue.append(i)
                # insert all unvisited neighbor points of the core point to ordered queue.
                self.__insert_ordered_queue(i)
                while self.ordered_queue:
                    # pop first point in ordered queue and add it to result queue.
                    nearest = self.ordered_queue.pop(0)
                    self.visited[nearest] = 1
                    self.result_queue.append(nearest)
                    # if the popped point is a core point,
                    # insert all its unvisited neighbor points to ordered queue.
                    if self.core_points[nearest] != 0:
                        self.__insert_ordered_queue(nearest)
        # push all left unvisited points(noise points) to result queue.
        for i in range(self.len):
            if self.visited[i] == 0:
                self.result_queue.append(i)
                self.reachable_distances[i] == -2

    def cluster_extract(self, Eps):
        ID = -1
        k = 1
        for i in range(self.len):
            # fetch the first point in result queue.
            p = self.result_queue[i]
            # if its rd is larger than given Eps, or it has no rd,
            if self.reachable_distances[p] > Eps or self.reachable_distances[p] == -1:
                # if it's a core point and its core distance smaller than given Eps,
                # it's the start of a new cluster.
                if self.core_distances[p] != 0 and self.core_distances[p] < Eps:
                    ID = k
                    k += 1
                    self.cluster_labels[p] = ID
                # it's a noise point.
                else:
                    self.cluster_labels[p] = -1
            # if its rd is -2, its a noise point.
            elif self.reachable_distances[p] == -2:
                self.cluster_labels[p] = -1
                # if its rd is within given Eps, it belongs to the same cluster.
            else:
                self.cluster_labels[p] = ID
