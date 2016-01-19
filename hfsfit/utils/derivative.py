# derivative.py - determine first and second derivative

import numpy as np


## ---- First derivative of y w.r.t x ----
def firstderivxy(x, y):
    """
    Function to determine first derivative of y w.r.t x (dy/dx).
    
    Parameters
    ----------
    x : ndarray
        
    y : ndarray
    
    Output
    ------
    ndarray
        first derivative
    
    Reference
    ---------
    Implementation of MATLAB code from
    http://terpconnect.umd.edu/~toh/spectrum/functions.html
    """
    n = y.shape[0]
    d = np.zeros(n)
    d[0] = (y[2] - y[1]) / (x[2] -x[1])
    d[n-1] = (y[n-1] - y[n-2]) / (x[n-1] -x[n-2])
    
    for i in range(1, n-1):
        d[i] = (y[i] - y[i-1]) / (2 * (x[i] -x[i-1]))
        
    return d
    
    
## ---- Second derivative of y w.r.t. x ----
def secderivxy(x, y):
    """
    Function to determine second derivative of y w.r.t x (dy/dx).
    
    Parameters
    ----------
    x : ndarray
        
    y : ndarray
    
    Output
    ------
    ndarray
        Second derivative
    
    Reference
    ---------
    Implementation of MATLAB code from
    http://terpconnect.umd.edu/~toh/spectrum/functions.html
    """    
    n = y.shape[0]
    d = np.zeros(n)
    
    for i in range(1, n-1):
        x1, x2, x3 = x[i-1], x[i], x[i+1]
        d[i] = ((y[i+1] - y[i]) / (x3 - x2) - (y[i] - y[i-1]) / (x2 - x1)) / ((x3 -x1)/2)
    
    d[0] = d[1]
    d[n-1] = d[n-2]
    
    return d


## ---- First derivative of signal ----
def firstderiv(signal):
    """
    Function to determine first derivative of signal using 2-point central
    difference.
    
    Parameters
    ----------
    signal : ndarray
        
    Output
    ------
    ndarray
        First derivative
    
    Reference
    ---------
    Implementation of MATLAB code from
    http://terpconnect.umd.edu/~toh/spectrum/functions.html
    """      
    n = signal.shape[0]
    d = np.zeros(n)
    
    d[0] = signal[1] - signal[0]
    d[n-1] = signal[n-1] - signal[n-2]
    
    for j in range(1, n-1):
        d[j] = (signal[j+1] - signal[j-1]) / 2.0
    
    return d


## ---- Second Derivative of signal ----
def secderiv(signal):
    """
    Function to determine second derivative of signal using 3-point central
    difference.
    
    Parameters
    ----------
    signal : ndarray
        
    Output
    ------
    ndarray
        Second derivative
    
    Reference
    ---------
    Implementation of MATLAB code from
    http://terpconnect.umd.edu/~toh/spectrum/functions.html
    """       
    # Second derivative of vector using 3-point central difference.
    n = signal.shape[0]
    d = np.zeros(n)
    
    for j in range(1, n-1):
        d[j] = signal[j+1] - 2 * signal[j] + signal[j]
    
    d[0] = d[1]
    d[n-1] = d[n-2]
    
    return d