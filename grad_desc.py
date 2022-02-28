#gradient descent
from CostFunction import costfunction
import numpy as np

def grad_descent(theta,X,Y,alpha,iterations):
    #theta = theta - alpha*gradient_vec
    #return theta
    itercost = []
    for i in range(iterations):
        (J, grd) = costfunction(theta,X,Y)
        itercost.append(J)
        theta = theta - alpha*grd
        
    costs = np.array(itercost,dtype=object)
    jcost = costs.reshape((iterations,1))
        
    return jcost, theta
        