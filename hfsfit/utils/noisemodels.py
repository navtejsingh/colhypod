# noisemodels.py - Random noise models

## ---- Import python modules ----
import numpy as np


__all__ = ['whitenoise', 'pinknoise', 'propnoise', 'sqrtnoise', 'bimodalnoise']


def whitenoise(x):
    """
    Function to generate white noise. Whitenoise is normal distributed noise 
    with 0 mean and 1 standard deviation. It is same at all wavelengths.
    
    Parameters
    ----------
    x : ndarray
        Input 1D numpy array
    
    Output
    ------
    ndarray
        Output white noise with zero mean and 1 standard deviation
    
    Reference
    ---------
    Implementation of MATLAB code from
    http://terpconnect.umd.edu/~toh/spectrum/noisetest.m
    """
    return np.random.randn(x.shape[0])


def pinknoise(x):
    pass
    

def propnoise(x):
    """
    Function to generate proportional random noise.
    
    Parameters
    ----------
    x : ndarray
        Input 1D numpy array
    
    Output
    ------
    ndarray
        Output proportional noise.
    
    Reference
    ---------
    Implementation of MATLAB code from
    http://terpconnect.umd.edu/~toh/spectrum/noisetest.m
    """    
    z = x * np.random.randn(x.shape[0])
    sz = np.std(z)
    return z / sz    


def sqrtnoise(x):
    """
    Function to generate square root random noise.
    
    Parameters
    ----------
    x : ndarray
        Input 1D numpy array
    
    Output
    ------
    ndarray
        Output square root random noise.
    
    Reference
    ---------
    Implementation of MATLAB code from
    http://terpconnect.umd.edu/~toh/spectrum/noisetest.m
    """     
    z = np.sqrt(np.abs(x)) * np.random.randn(x.shape[0])
    sz = np.std(z)
    return z / sz        
    
    
def bimodalnoise(x, a, b, std):
    """
    Function to generate bimodal random noise.
    
    Parameters
    ----------
    x : ndarray
        Input 1D numpy array
    
    a : float
        First peak position
    
    b : float
        Second peak position
    
    std: float
        Standard deviation
    
    Output
    ------
    ndarray
        Output bimodal random noise.
    
    Reference
    ---------
    Implementation of MATLAB code from
    http://terpconnect.umd.edu/~toh/spectrum/noisetest.m
    """       
    lx = x.shape[0]
    z = np.zeros(lx)
    
    j = 0
    while j < lx:
        if np.random.rand() < 0.5:
            z[j] = a + std * np.random.randn()
        else:
            z[j] = b + std * np.random.randn()
            
        j += 1
    
    return z