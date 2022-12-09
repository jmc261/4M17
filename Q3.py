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

# Set up a number of the other parameters
m = x_0.shape[0] # =256
n = b.shape[0] # = 60
t = 1
lambda_max = np.linalg.norm(2*np.dot(A.T, b), ord=np.inf)


"""
def f(x):
    # Test function for Newton Method

    total = 0
    for x_i in x:
        total += x_i**4
    
    return total

def grad_f(x):
    # Test function for Newton method

    grad_f_x = np.zeros((x.shape[0], 1))
    for i, x_i in enumerate(x):
        #print(grad_f_x.shape)
        grad_f_x[i] = 4*x_i**3

    return grad_f_x

def Hess_f(x):
    # Test function for Newton method

    Hess_f_x = np.zeros((x.shape[0], x.shape[0]))

    for i, x_i in enumerate(x):
        Hess_f_x[i, i] = 12*x_i**2

    return Hess_f_x
"""
def f(x_u, t, lambda_max=lambda_max, m=m, A=A, b=b):
    # Calculate f, given by Eq. (5)

    l = 0.01 * lambda_max
    f_x = 0

    # Add the f_0 terms
    f_x += t * np.linalg.norm((np.dot(A, x_u[:m])-b), ord=2)**2
    f_x += t * l * np.sum(x_u[m:])
    
    # Add the contributions from the barrier terms
    f_1 = x_u[:m] + x_u[m:] # x + u
    f_2 = -x_u[:m] + x_u[m:] # -x + u

    # Return infinity if either barrier term is negative. This is
    # because np.log just returns Nan, so won't blow up like it should
    if ((f_1 <= 0).any()) or ((f_2 <= 0).any()):
        return np.inf

    for j in range(m):
        f_x -= np.log(f_1[j])
        f_x -= np.log(f_2[j])

    return f_x

def grad_f(x_u, t, lambda_max=lambda_max, m=m, A=A, b=b):
    # Returns the gradient of the curve

    # Initialise the vector
    grad_f_x = np.zeros((2*m, 1))
    
    l = 0.01*lambda_max
    Ax_minus_b = np.dot(A, x_u[:m])-b
    # Find the barrier terms
    f_1 = x_u[:m] + x_u[m:] # x + u
    f_2 = -x_u[:m] + x_u[m:] # -x + u

    # Input the values corresponding to the x vector
    for p in np.arange(m):
        for k in np.arange(n):
            grad_f_x[p] += 2*t*A[k, p]*Ax_minus_b[k]
        # Subtract the barrier terms
        grad_f_x[p] -= 1/f_1[p]
        grad_f_x[p] += 1/f_2[p]
    
    # Input the values corresponding to the u vector
    for i in np.arange(m):
        grad_f_x[i+m] += t*l
        grad_f_x[i+m] -= 1/(f_1[i])
        grad_f_x[i+m] -= 1/(f_2[i])

    return grad_f_x

def Hess_f(x_u, t, m=m, A=A, b=b):

    # Build the Hess x, x matrix, without the barrier terms
    Hess_xx = np.zeros((m, m))
    # Add the 'non-barrier' terms
    for p in range(m):
        for q in range(m):
            for k in range(n):
                Hess_xx[q, p] += 2* t * A[k, p] * A[k, q]

    # Calculate the barrier terms
    f_1 = x_u[:m] + x_u[m:] # x + u
    f_2 = -x_u[:m] + x_u[m:] # -x + u

    # Build the Hess x, u matrix
    Hess_xu_diagonal = np.divide(np.ones((1, m)), np.power(f_1, 2))
    Hess_xu_diagonal -= np.divide(np.ones((1, m)), np.power(f_2, 2))
    # Include [0] as Hess_xu is in a (1, 256) array
    Hess_xu = np.diag(Hess_xu_diagonal[0])

    # Build the full Hessian matrix
    left_Hess = np.vstack([Hess_xx, Hess_xu])
    right_Hess = np.vstack([Hess_xu, np.zeros((m, m))])
    Hess = np.hstack([left_Hess, right_Hess]) # Combine to form one large matrix

    # Now need to add the 'barrier terms'
    Hess_diagonal = np.divide(np.ones((1, m)), np.power(f_1, 2))
    Hess_diagonal += np.divide(np.ones((1, m)), np.power(f_2, 2))

    Hess_diagonal = np.append(Hess_diagonal, Hess_diagonal)
    Hess = Hess + np.diag(Hess_diagonal)

    return Hess

def calc_newton_vars(x_u, t, f=f, grad_f=grad_f, Hess_f=Hess_f):
    # Calculates the newton step size and decrement

    # Find gradient and Hessian
    grad_f_x = grad_f(x_u, t)
    Hess_inv = np.linalg.inv(Hess_f(x_u, t))
    """
    # Inverse of Hessian is very expensive to compute, so use Chol Decomposition
    L = np.linalg.cholesky(Hess_f_x)
    L_inv = np.linalg.inv(L)

    # Find Newton variables
    newton_step = -np.dot(np.dot(L_inv.T, L_inv), grad_f_x)
    lambda_squared = np.linalg.norm(np.dot(L_inv.T, grad_f_x), ord=2)
    """
    # Find Newton variables
    newton_step = -np.dot(Hess_inv, grad_f_x)
    lambda_squared = np.dot(grad_f_x.T, -newton_step)[0][0]
    
    return newton_step, lambda_squared

def backtrack_iter(x_un, step_dir, alpha, beta, convergence_check):
    # Performs one full iteration (i.e finds a suitable 'minimum' for a given descent direction)

    # Calculate all values that are not unique to each loop
    tau = 1
    f_x = f(x_un[-1, :], t)
    grad_f_x = grad_f(x_un[-1, :], t)
    """
    print(f"grad_f_x = {np.dot(grad_f_x.T, step_dir)}")
    print(f"grad_f = {np.linalg.norm(grad_f_x, ord=2)}")
    print(f_x)
    """
    #print(f"1 = {f(x_un[-1, :] + tau*step_dir.T[0], t)}, 2 = {f_x}, 3 = {(alpha * tau * np.dot(grad_f_x.T, step_dir))[0][0]}")
    # Reduce tau until criteria is met
    while f(x_un[-1, :] + tau*step_dir.T[0], t) >= f_x + (alpha * tau * np.dot(grad_f_x.T, step_dir))[0][0]:
        tau *= beta

    #print(f"difference: {f(x_un[-1, :], t) - f(x_un[-1, :] + tau*step_dir.T[0], t)}")

    #In case the line search passes over the minimum
    if f(x_un[-1, :] + tau*step_dir.T[0], t) > f_x:
        x_un = np.vstack([ x_un, x_un[-1, :]])
        convergence_check = np.append(convergence_check, f_x)
    else:
        # Update xn and norm matrices
        x_un = np.vstack([ x_un, (x_un[-1, :] + tau*step_dir.T[0]) ])
        #l1_norms = np.append(l1_norms, np.linalg.norm( np.dot(A, xn[-1, :n]) - b, ord=1 ))
        convergence_check = np.append(convergence_check, f(x_un[-1, :], t))

    return x_un, convergence_check

# Set up initial x_u vector, with initial values
# These are the values that will be recorded (i.e after each Newton iteration)
x_u_rec = np.array([np.append(np.ones((m, 1))/2, np.ones((m, 1)))])

tol_newton = 0.001
tol_back = 0.001

alpha = 0.5
beta = 0.5

for _ in range(10):
    print(t)
    # Perform first iteration outside the loop
    newton_step, lambda_squared = calc_newton_vars(x_u_rec[-1, :], t)
    while lambda_squared/2 >= tol_newton:
        # Initialise vectors
        # These values (except the final, converged value) will not be saved
        x_un = np.array([x_u_rec[-1, :]])
        convergence_check = np.array([f(x_un[-1, :], t)])

        # Run first iteration out of loop, to enable convergence criteria
        x_un, convergence_check = backtrack_iter(x_un, newton_step, alpha, beta, convergence_check)

        # Repeat until backtracking has converged
        # Since we are providing a fixed direction, if a step is too big and it passes over
        # the minimum, without being within the convergence limits, the method will continue
        # past the minimum, so add another check to prevent that
        while np.abs(convergence_check[-2] - convergence_check[-1]) > tol_back:
            #print(np.abs(convergence_check[-2] - convergence_check[-1]))
            x_un, convergence_check = backtrack_iter(x_un, newton_step, alpha, beta, convergence_check)
    
        # Update x
        x_u_rec = np.vstack([x_u_rec, x_un[-1, :]])
        #print(f(x_u_rec[-1, :], t))

        # Find new Newton parameters
        newton_step, lambda_squared = calc_newton_vars(x_u_rec[-1, :], t)

    t *= 5

print(f"{x_u_rec.shape[0]} iterations")

# Plot the results

x = x_u_rec[-1, :m]

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.bar(np.arange(m), x)
ax2.bar(np.arange(m), x_0)
plt.show()


