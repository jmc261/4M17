import numpy as np
from matplotlib import pyplot as plt
from functions import *

if __name__ == '__main__':
    n = 1000
    # Form a grid
    x_1 = np.linspace(-500, 500, n)
    X_1, X_2 = np.meshgrid(x_1, x_1)
    coords = zip(X_1, X_2)
    print(X_1.shape)

    # Evaluate f
    f = np.array([])
    for coord in coords:
        f = np.append(f, schwefel(coord))

    f = f.reshape(n, n)

    # 3D Plot Schwefel
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(X_1, X_2, f, cmap='rainbow')
    plt.show()