import numpy as np
from sigmoid import sigmoid

#theta and Y must be column vectors(arrays).
#X must be such that each row is a training example.

def costfunction(theta,X,Y):
    [n,m] = X.shape
    h = sigmoid(X@theta)
    p1 = np.transpose(-Y)@np.log(h)
    p2 = np.transpose(1-Y)@np.log(1-h)
    J = (1/n)*(p1-p2)
    
    grd = (1/n)*((np.transpose(X))@(h-Y))
    
    return J,grd

    
#This works(so far...).