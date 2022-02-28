import numpy as np

#input has to be an array

def sigmoid(z):
    g = 1/(1 + np.exp(-z))
    return g

#This is fine.