import numpy as np
import scipy
from matplotlib import pyplot as plt
import pandas as pd
from timeit import default_timer as timer
import seaborn as sns

def f(x):
    # Test function for Newton Method

    total = 0
    for x_i in x:
        total += x_i**4 + 2.4*x_i**3
    
    return total

def grad_f(x):
    # Test function for Newton method

    grad_f_x = np.zeros((x.shape[0], 1))
    for i, x_i in enumerate(x):
        #print(grad_f_x.shape)
        grad_f_x[i] = 4*x_i**3 + 7.2*x_i**2

    return grad_f_x

def Hess_f(x):
    # Test function for Newton method

    Hess_f_x = np.zeros((x.shape[0], x.shape[0]))

    for i, x_i in enumerate(x):
        Hess_f_x[i, i] = 12*x_i**2 + 14.4*x_i

    return Hess_f_x

def calc_newton_vars(x, f=f, grad_f=grad_f, Hess_f=Hess_f):
    # Calculates the newton step size and decrement

    # Find gradient and Hessian
    grad_f_x = grad_f(x)
    Hess_inv = np.linalg.inv(Hess_f(x))

    # Find newton variables
    newton_step = -np.dot(Hess_inv, grad_f_x)
    lambda_squared = np.dot(grad_f_x.T, -newton_step)[0][0]

    return newton_step, lambda_squared

def backtrack_iter(xn, step_dir, alpha, beta, convergence_check):
    # Performs one full iteration (i.e finds a suitable 'minimum' for a given descent direction)

    # Calculate all values that are not unique to each loop
    tau = 1
    f_x = f(xn[-1, :])
    grad_f_x = grad_f(xn[-1, :])

    # Reduce tau until criteria is met
    while f(xn[-1, :] + tau*step_dir)[0] >= f_x + (alpha * tau * np.dot(grad_f_x.T, step_dir)):
        tau *= beta

    # Update xn and norm matrices
    xn = np.vstack([ xn, (xn[-1, :] + tau*step_dir) ])
    #l1_norms = np.append(l1_norms, np.linalg.norm( np.dot(A, xn[-1, :n]) - b, ord=1 ))
    convergence_check = np.append(convergence_check, f(xn[0, :]))

    return xn, convergence_check

x = np.array([[-1, -1, -1]])

tol_newton = 0.001
tol_back = 0.001

alpha = 0.5
beta = 0.5

# Perform first iteration outside the loop
newton_step, lambda_squared = calc_newton_vars(x[0, :])

print(newton_step)
print(lambda_squared/2)
print(tol_newton)

while lambda_squared/2 >= tol_newton:
    print("here")
    # Initialise vectors
    xn = np.array([x[-1, :]])
    convergence_check = np.array([f(xn[0, :])])

    # Run first iteration out of loop, to enable convergence criteria
    xn, convergence_check = backtrack_iter(xn, newton_step, alpha, beta, convergence_check)

    # Repeat until backtracking has converged
    while np.abs(convergence_check[-2] - convergence_check[-1]) > tol_back:
        xn, convergence_check = backtrack_iter(xn, newton_step, alpha, beta, convergence_check)
    
    # Update x
    x = np.vstack([x, xn[-1, :]])

    # Find new Newton parameters
    newton_step, lambda_squared = calc_newton_vars(x[-1, :])
    print(xn.shape)

#print(x.shape)
#print(x[-1, :])


