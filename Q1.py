import numpy as np
import scipy
from matplotlib import pyplot as plt
import pandas as pd
import csv

A_csvs = [ "A1", "A2", "A3", "A4", "A5" ]
b_csvs = [ "b1", "b2", "b3", "b4", "b5" ]
ns = [ 16, 64, 256, 512, 1024 ]

index = 1

# Extract the A matrix from the csv file
A = np.genfromtxt(f"C:/Users/jmc261/git/4M17/data/Q1/{A_csvs[index-1]}.csv", delimiter=',')

# Extract the b vector from the csv file
b = np.genfromtxt(f"C:/Users/jmc261/git/4M17/data/Q1/{b_csvs[index-1]}.csv", delimiter=',')

print(b.shape)