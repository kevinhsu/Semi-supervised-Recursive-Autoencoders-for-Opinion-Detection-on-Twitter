__author__ = 'Yazhe'

import numpy as np
import warnings
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
    return x*(1-x)

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def tanh_prime(x):
    return 1-x*x

def norm(x):
    if np.ndim(x)==1:
        return np.sqrt(np.sum(x**2))
    else:
        return np.sqrt(np.sum(x**2,1))

def norm1tanh_prime(x):
    y=x-x**3
    nrm=norm(x)
    return np.diag(1-x**2)/nrm-np.outer(y,x)/(nrm**3)

def softmax(x):
    vec=np.exp(x)
    denom=np.sum(vec)
    return vec/denom

def softmax_prime(x):
    return np.diag(x)-np.outer(x,x)