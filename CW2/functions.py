import numpy as np

def schwefel(x):
    ### Returns the function evaluation of an N-Dimensional Schwefel function

    f = 0
    for x_i in x:
        f -= x_i * np.sin(np.sqrt(np.abs(x_i)))
    
    return f

