import numpy as np
import scipy
from matplotlib import pyplot as plt
import pandas as pd
from timeit import default_timer as timer
import seaborn as sns

def find_l1_opt(A, b):
    # Creates the 'tilde' matrices and solves the LP to find the l1 optimum

    m, n = A.shape
    # Form the 'tilde' matrices
    stacked_As = np.vstack([A, -A])
    stacked_Is = np.vstack([-np.identity(m), -np.identity(m)])
    A_tilde = np.hstack([stacked_As, stacked_Is])

    b_tilde = np.hstack([b, -b])

    c_tilde = np.vstack([np.zeros((n, 1)), np.ones((m, 1))])

    start_time = timer()
    # Use scipy to find optimum in l1 norm
    optimised_results = scipy.optimize.linprog(c_tilde, A_ub = A_tilde, b_ub = b_tilde)
    end_time = timer()

    # Find optimum x - note that the output of the LP is x_tilde
    x_tilde = optimised_results['x']
    x_opt = x_tilde[:n]

    # Find the residuals, and the corresponding norm
    r = np.dot(A, x_opt) - b
    opt_l1_norm = np.linalg.norm(r, ord=1)

    return x_opt, opt_l1_norm, r, end_time-start_time

def find_l2_opt(A, b):
    # Solve the optimisation analytically and find the minimised norm

    start_time = timer()
    # Find optimum x
    x_opt = np.linalg.lstsq(A, b)[0]
    end_time = timer()

    # Find the residuals, and the corresponding norm
    r = np.dot(A, x_opt) - b
    opt_l2_norm = np.linalg.norm(r, ord=2)

    return x_opt, opt_l2_norm, r, end_time - start_time

def find_linf_opt(A, b):
    # Creates the 'tilde' matrices and solves the LP to find the l1 optimum

    m, n = A.shape
    # Form the 'tilde' matrices
    A_tilde = np.hstack([ np.vstack([A, -A]), -np.ones((2*m, 1)) ])

    b_tilde = np.hstack([b, -b])

    c_tilde = np.vstack([np.zeros((n, 1)), np.array([1])])

    start_time = timer()
    # Use scipy to find optimum in l_inf norm
    optimised_results = scipy.optimize.linprog(c_tilde, A_ub = A_tilde, b_ub = b_tilde)
    end_time = timer()

    # Find optimum x - note that the output of the LP is x_tilde
    x_tilde = optimised_results['x']
    x_opt = x_tilde[:-1]

    # Find the residuals, and the corresponding norm
    r = np.dot(A, x_opt) - b
    opt_linf_norm = np.linalg.norm(r, ord=np.inf)

    return x_opt, opt_linf_norm, r, end_time-start_time

A_csvs = [ "A1", "A2", "A3", "A4", "A5" ]
b_csvs = [ "b1", "b2", "b3", "b4", "b5" ]

A_csvs = ["A3"]
b_csvs = ["b3"]

# Create an array to hold the results
results = np.zeros((len(A_csvs), 6))
"""df = pd.DataFrame(data = np.zeros((len(A_csvs), 6)),
                        rows = [1, 2, 3, 4, 5]
                        columns = [r"l_1 norm", "l_1 runtime", r"l_2 norm",
                        "l_2 runtime", r"l_inf norm", "l_inf runtime"])"""

starting_time = timer()

for index, Acsv in enumerate(A_csvs):

    # Extract the A matrix from the csv file
    A = np.genfromtxt(f"C:/Users/jmc261/git/4M17/data/Q1/{A_csvs[index-1]}.csv", delimiter=',')
    # Extract the b vector from the csv file
    b = np.genfromtxt(f"C:/Users/jmc261/git/4M17/data/Q1/{b_csvs[index-1]}.csv", delimiter=',')
    m, n = A.shape # Find the matrix dimensions

    x_opt_l1, l1_norm, l1_residuals, l1_time_taken = find_l1_opt(A, b)
    results[index, 0] = l1_norm
    results[index, 1] = l1_time_taken

    x_opt_l2, l2_norm, l2_residuals, l2_time_taken = find_l2_opt(A, b)
    results[index, 2] = l2_norm
    results[index, 3] = l2_time_taken
    
    x_opt_linf, linf_norm, linf_residuals, linf_time_taken = find_linf_opt(A, b)
    results[index, 4] = linf_norm
    results[index, 5] = linf_time_taken
    
    print(f"Finished {index+1}")

print(f"Norm is{np.linalg.norm(x_opt_l1, ord=2)}")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

results_table = pd.DataFrame(data=results,
                            columns = [r"l_1 norm", "l_1 runtime", r"l_2 norm",
                            "l_2 runtime", r"l_inf norm", "l_inf runtime"])

ending_time = timer()

print(results_table)
print(ending_time-starting_time)

sns.histplot(data=l1_residuals, ax=ax1)
sns.histplot(data=l2_residuals, ax=ax2)
sns.histplot(data=linf_residuals, ax=ax3)

plt.show()