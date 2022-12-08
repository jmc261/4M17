import numpy as np
import scipy
from matplotlib import pyplot as plt
import pandas as pd
from timeit import default_timer as timer
import seaborn as sns

"""
def f(x):
    # Test function - just sum of squares
    return np.dot(x, x)

def grad_f(x):
    # Test function - gradient of f(x)
    return 2*x
"""

# Extract the A matrix from the csv file
A = np.genfromtxt(f"C:/Users/jmc261/git/4M17/data/Q1/A3.csv", delimiter=',')
# Extract the b vector from the csv file
b = np.genfromtxt(f"C:/Users/jmc261/git/4M17/data/Q1/b3.csv", delimiter=',')

# Form the 'tilde' matrices
m, n = A.shape
stacked_As = np.vstack([A, -A])
stacked_Is = np.vstack([-np.identity(m), -np.identity(m)])
A_tilde = np.hstack([stacked_As, stacked_Is])

b_tilde = np.hstack([b, -b])

c_tilde = np.append(np.zeros((n, 1)), np.ones((m, 1)))

def f(x_tilde, t, A_tilde = A_tilde, b_tilde = b_tilde, c_tilde = c_tilde):
    # Calculate objective function

    # Also returns residuals as they are useful for
    # calculating grad_f (and grad_f is only called if f is known already)

    # Calculate Ax - b (<= 0)
    f_i_vector = np.dot(A_tilde, x_tilde).T - b_tilde
    
    # each f_i should be negative (so -f_i is positive)
    # if x is out of bounds, return 'infinity'
    if (f_i_vector >= 0).any() == True:
        return np.inf, f_i_vector

    # Calculate objective function using formula in report
    total = t * np.dot(c_tilde, x_tilde)
    for f_i in f_i_vector:
        total -= np.log(-f_i)

    return total, f_i_vector

def grad_f(f_i_vector, t, A_tilde = A_tilde, c_tilde = c_tilde):
    # Finds grad f. Uses f_i from f(x), since grad_f is only ever called after it

    total = t * c_tilde

    for i, f_i in enumerate(f_i_vector):
        total -= A_tilde[i, :]/f_i

    return total

def perform_iter(f, grad_f, beta, alpha, xn, l1_norms, convergence_norms):
    # Performs one full iteration (i.e finds a suitable 'minimum' for a given descent direction)

    # Calculate all values that are not unique to each loop
    tau = 1
    f_x, f_i_vector = f(xn[-1, :], 1)
    grad_f_x = grad_f(f_i_vector, 1)
    step_dir = - grad_f_x  # choose grad_f as step direction, delta_x in notes

    # Reduce tau until criteria is met
    while f(xn[-1, :] + tau*step_dir, 1)[0] >= f_x + (alpha * tau * np.dot(grad_f_x, step_dir)):
        tau *= beta

    # Update xn and norm matrices
    xn = np.vstack([ xn, (xn[-1, :] + tau*step_dir) ])
    l1_norms = np.append(l1_norms, np.linalg.norm( np.dot(A, xn[-1, :n]) - b, ord=1 ))
    convergence_norms = np.append(convergence_norms, np.linalg.norm(np.dot(A_tilde, xn[-1, :])-b_tilde, ord=1))

    #print(f(xn[-1, :], 1)[0])

    return xn, l1_norms, convergence_norms

start = timer()

# Parameters for backwards line search
beta = 0.5
alpha = 0.5
t = 1

# Matrix which stores previous iterations
# Initialise with c_tilde
xn = np.array([c_tilde])
# Vector storing l1 norm of (Ax - b) i.e the desired minimisation
# note it does not contain norm of (A_tilde*x_tilde - b_tilde)
l1_norms = np.array([np.linalg.norm( (np.dot(A, xn[0, :n]) - b), ord=1 )])

# Vector storing l1 norm of (A_tilde*x_tilde - b_tilde), used for convergence criteria
convergence_norms = np.array([np.linalg.norm( np.dot(A_tilde, xn[0, :]) - b_tilde, ord=1 )])

# Technically, perform_iter leads to better performance if the function itself is just pasted here,
# but perform_iter has been created for clarity and readability

# Perform the first iteration outside of the while loop (allows the convergence criteria to be checked)
xn, l1_norms, convergence_norms = perform_iter(f, grad_f, beta, alpha, xn, l1_norms, convergence_norms)

# Convergence criteria - keep iterating until the function changes very little between iterations
while np.abs(convergence_norms[-2] - convergence_norms[-1]) > 0.001:
    #print(convergence_norms[-2] - convergence_norms[-1])
    xn, l1_norms, convergence_norms = perform_iter(f, grad_f, beta, alpha, xn, l1_norms, convergence_norms)

end = timer()
print(end-start)

# Plot evolution of l1_norms vs iteration

print("We here")

# Find the number of iterations, create x values
no_iters = l1_norms.shape[0]
x = np.arange(no_iters) + 1

# Add data to a pandas dataframe, so seaborn can be used to plot
df = pd.DataFrame(np.column_stack([x, l1_norms]), columns = ["Iteration", "l1_norm"])

# Set up a semi-log scale
fig, ax1 = plt.subplots()
ax1.set(xscale='log', yscale='linear')

sns.lineplot(data=df, x="Iteration", y="l1_norm", ax=ax1)

plt.show()