# gaussfit.py - Routines to fit single Gaussian peaks

## ---- Import python routines ----
import sys
import numpy as np


## ---- Caruana's linear least square fitting ----  
def caruana(x, y):
    """
    Linear least square fitting routine to fit a quadratic polynomial to
    Gaussian peak. Parabola is fitted to (x,lny) to determine fitted values
    of height, position, and FWHM of Gaussian peak.
    
    Parameters
    ----------
    x : ndarray
        x-axis data
        
    y : ndarray
        y-axis data
        
    Returns
    -------
    height, position, width : float
        height, position and FWHM of fitted Gaussian peaki
    
    
    Reference
    ---------
    Routine is based on Caruana algorithm mention in paper by Hongwei Guo.
    'A Simple Algorithm for Fitting a Gaussian Function', IEEE Signal
    Processing Magazine, September 2011
    """
    # Common variables
    N = len(x)
    Sx = np.sum(x)
    Sx2 = np.sum(x**2)
    Sx3 = np.sum(x**3)
    Sx4 = np.sum(x**4)
    
    # To avoid undefined ln(y)
    maxy =np.max(y)
    ind = np.arange(len(y))
    mody = np.array(map(lambda i: maxy/100.0 if y[i] < maxy/100.0 else y[i], ind)) 
    
    # Create a 3x3 matrix and a 3 dimensional vector
    X = np.array([[N, Sx, Sx2], [Sx, Sx2, Sx3], [Sx2, Sx3, Sx4]])
    Y = np.vstack(np.array([np.sum(np.log(mody)), np.sum(x * np.log(mody)), np.sum(x**2 * np.log(mody))]))
    
    # Solve linear equation using least square
    a, b, c = np.linalg.lstsq(X,Y)[0]
    
    # Calculate fitted values of height, position and width
    height = np.exp(a - ((b**2) / (4 * c)))
    position = -b/(2*c)
    width = 1.66511 / np.sqrt(-c)
    
    return np.array([height, position, width])


## ---- Weighted linear least square fitting ----  
def wlstsq(x, y):
    """
    Weighted Linear least square fitting routine to fit a quadratic polynomial 
    to Gaussian peak. Parabola is fitted to (x,lny) to determine fitted values 
    of height, position and fwhm of Gaussian peak.
    
    Parameters
    ----------
    x : ndarray
        x-axis data
        
    y : ndarray
        y-axis data
        
    Returns
    -------
    height, position, width : float
        height, position and FWHM of fitted Gaussian peak
    
    References
    ----------
    Routine is based on weighted algorithm mention in paper by Hongwei Guo.
    'A Simple Algorithm for Fitting a Gaussian Function', IEEE Signal
    Processing Magazine, September 2011
    """
    # To avoid undefined ln(y)
    maxy =np.max(y)
    ind = np.arange(len(y))
    mody = np.array(map(lambda i: maxy/100.0 if y[i] < maxy/100.0 else y[i], ind))     
   
    # Common variables
    Sy2 = np.sum(mody**2)
    Sxy2 = np.sum(x * mody**2)
    Sx2y2 = np.sum(x**2 * mody**2)
    Sx3y2 = np.sum(x**3 * mody**2)
    Sx4y2 = np.sum(x**4 * mody**2)
    
    # Create a 3x3 matrix and a 3 dimensional vector
    X = np.array([[Sy2, Sxy2, Sx2y2], [Sxy2, Sx2y2, Sx3y2], [Sx2y2, Sx3y2, Sx4y2]])
    Y = np.vstack(np.array([np.sum(mody**2 * np.log(mody)), np.sum(x * mody**2 * np.log(mody)), np.sum(x**2 * mody**2 * np.log(mody))]))
    
    # Solve linear equation using least square
    a, b, c = np.linalg.lstsq(X,Y)[0]
     
    # Calculate fitted values of height, position and width
    height = np.exp(a - ((b**2) / (4 * c)))
    position = -b/(2*c)
    width = 1.66511 / np.sqrt(-c)
    
    return np.array([height, position, width])
    
    
    
## ---- Iterative Weighted linear least square fitting ----  
def witerlstsq(x, y, niter = 10):
    """
    Weighted iterative Linear least square fitting routine to fit a quadratic 
    polynomial to Gaussian peak. Parabola is fitted to (x,lny) to determine 
    fitted values of height, position and fwhm of Gaussian peak.
    
    Parameters
    ----------
    x : ndarray
        x-axis data
        
    y : ndarray
        y-axis data
        
    niter : int
        Number of iterations. Default is 10.
        
    Returns
    -------
    height, position, width : float
        height, position and FWHM of fitted Gaussian peak
    
    References
    ----------
    Routine is based on weighted algorithm mention in paper by Hongwei Guo.
    'A Simple Algorithm for Fitting a Gaussian Function', IEEE Signal
    Processing Magazine, September 2011
    """    
    # To avoid undefined ln(y)
    maxy =np.max(y)
    ind = np.arange(len(y))
    mody = np.array(map(lambda i: maxy/100.0 if y[i] < maxy/100.0 else y[i], ind))     
   
    # Common variables
    a, b, c = np.zeros(1), np.zeros(1), np.zeros(1)
    for k in range(niter):
        print >> sys.stdout, '\n Iteration - ', k+1
        
        if k == 0:
            moditery = np.copy(mody)
        else:
            moditery = np.exp(a + b * x + c * x**2)    
                        
        Sy2 = np.sum(moditery**2)
        Sxy2 = np.sum(x * moditery**2)
        Sx2y2 = np.sum(x**2 * moditery**2)
        Sx3y2 = np.sum(x**3 * moditery**2)
        Sx4y2 = np.sum(x**4 * moditery**2)
    
        # Create a 3x3 matrix and a 3 dimensional vector
        X = np.array([[Sy2, Sxy2, Sx2y2], [Sxy2, Sx2y2, Sx3y2], [Sx2y2, Sx3y2, Sx4y2]])
        Y = np.vstack(np.array([np.sum(moditery**2 * np.log(mody)), np.sum(x * moditery**2 * np.log(mody)), np.sum(x**2 * moditery**2 * np.log(mody))]))
    
        # Solve linear equation using least square
        a, b, c = np.linalg.lstsq(X,Y)[0]
     
        # Calculate fitted values of height, position and width
        height = np.exp(a - ((b**2) / (4 * c)))
        position = -b/(2*c)
        if c < 0:
            width = 1.66511 / np.sqrt(-c)
        else:
            width = - 1.66511 / np.sqrt(c)
    
    return np.array([height, position, width])