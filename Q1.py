import numpy as np
import scipy
from matplotlib import pyplot as plt
import pandas as pd
import csv

def find_l1_opt(A, b):
    # Creates the 'tilde' matrices and solves the LP to find the l1 optimum

    m, n = A.shape
    # Form the 'tilde' matrices
    stacked_As = np.vstack([A, -A])
    stacked_Is = np.vstack([-np.identity(m), -np.identity(m)])
    A_tilde = np.hstack([stacked_As, stacked_Is])

    b_tilde = np.hstack([b, -b])

    c_tilde = np.vstack([np.zeros((n, 1)), np.ones((m, 1))])

    # Use scipy to find optimum in l1 norm
    optimised_results = scipy.optimize.linprog(c_tilde, A_ub = A_tilde, b_ub = b_tilde)

    # Find optimum x - note that the output of the LP is x_tilde
    x_tilde = optimised_results['x']
    x_opt = x_tilde[:n]

    # Find norm of optimum x
    opt_linf_norm = np.linalg.norm((np.dot(A, x_opt) - b), ord=1)

    return x_opt, opt_linf_norm

def find_linf_opt(A, b):
    # Creates the 'tilde' matrices and solves the LP to find the l1 optimum

    m, n = A.shape
    # Form the 'tilde' matrices
    A_tilde = np.hstack([ np.vstack([A, -A]), -np.ones((2*m, 1)) ])

    b_tilde = np.hstack([b, -b])

    c_tilde = np.vstack([np.zeros((n, 1)), np.array([1])])

    # Use scipy to find optimum in l_inf norm
    optimised_results = scipy.optimize.linprog(c_tilde, A_ub = A_tilde, b_ub = b_tilde)

    # Find optimum x - note that the output of the LP is x_tilde
    x_tilde = optimised_results['x']
    x_opt = x_tilde[:-1]

    # Find norm of optimum x
    opt_linf_norm = np.linalg.norm((np.dot(A, x_opt) - b), ord=np.inf)

    return x_opt, opt_linf_norm

A_csvs = [ "A1", "A2", "A3", "A4", "A5" ]
b_csvs = [ "b1", "b2", "b3", "b4", "b5" ]

# Create an array to hold the results
results = np.zeros((len(A_csvs), 6))
"""df = pd.DataFrame(data = np.zeros((len(A_csvs), 6)),
                        rows = [1, 2, 3, 4, 5]
                        columns = [r"l_1 norm", "l_1 runtime", r"l_2 norm",
                        "l_2 runtime", r"l_inf norm", "l_inf runtime"])"""

for index, Acsv in enumerate(A_csvs):

    # Extract the A matrix from the csv file
    A = np.genfromtxt(f"C:/Users/jamie/git/4M17/data/Q1/{A_csvs[index-1]}.csv", delimiter=',')
    # Extract the b vector from the csv file
    b = np.genfromtxt(f"C:/Users/jamie/git/4M17/data/Q1/{b_csvs[index-1]}.csv", delimiter=',')
    m, n = A.shape # Find the matrix dimensions

    x_opt_l1, l1_norm = find_l1_opt(A, b)
    results[index, 0] = l1_norm

    x_opt_linf, linf_norm = find_linf_opt(A, b)
    results[index, 4] = linf_norm

results_table = pd.DataFrame(data=results,
                            columns = [r"l_1 norm", "l_1 runtime", r"l_2 norm",
                            "l_2 runtime", r"l_inf norm", "l_inf runtime"])

print(results_table)

    