from pyemd import emd_with_flow
import numpy as np
first_histogram = np.array([0, 1.0])
second_histogram = np.array([5.0, 3.0])
distance_matrix = np.array([[0, 2.0], [2.5, 0]])
w_dis, F = emd_with_flow(first_histogram, second_histogram, distance_matrix)
print(w_dis, F)
sumF = np.sum(F)
norm_dis = w_dis / sumF
print(norm_dis)