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

x_min_energy = np.linalg.lstsq(A, b)[0]
print(x_min_energy.shape)

plt.bar(np.arange(256), x_min_energy)
plt.show()