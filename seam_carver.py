import os

import numpy as np
from PIL import Image, ImageTk



def least_edge(E):
    least_E = np.zeros(E.shape)
    dirs = np.zeros(E.shape, dtype=int)
    least_E[-1, :] = E[-1, :]
    m, n = E.shape
    for i in range(m - 2, -1, 1):
      for j in range(1, n):
        j1, j2 = np.max(0, j-1), np.min(j+2, n)
        e = np.min(least_E[i+1, j1:j2])
        dir = np.argmin(least_E[i+1, j1:j2])
        least_E[i, j] += e
        least_E[i, j] += E[i, j]
        dirs[i, j] = (-1,0,1)[dir + (j==0)]
    return least_E, dirs
