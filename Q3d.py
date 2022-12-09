import numpy as np
import scipy
from matplotlib import pyplot as plt
import pandas as pd
from timeit import default_timer as timer
import seaborn as sns

# Extract the A matrix from the csv file
A = np.genfromtxt(f"C:/Users/jamie/git/4M17/data/Q3/A.csv", delimiter=',')
# Extract the b vector from the csv file
x_0 = np.genfromtxt(f"C:/Users/jamie/git/4M17/data/Q3/x0.csv", delimiter=',')

# Evaluate b
b = np.dot(A, x_0)

# Find least squares solution
x_min_energy = np.linalg.lstsq(A, b)[0]

#Plot the results
fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.bar(np.arange(256), x_min_energy)
ax1.set_ylim([-1, 1])
ax2.bar(np.arange(256), x_0)
plt.show()