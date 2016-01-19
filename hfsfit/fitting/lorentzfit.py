# lorentzfit.py - Linear Lorentzian peak fitting routines

import sys
import numpy as np

## ---- Numpy polyfit function ----    
def polyfit(x, y):
    """
    Use numpy polyfit to estimate height, width and position of the
    Lorentzian peak.
    
    Parameters
    ----------
    x : ndarray
        x-axis data
        
    y : ndarray
        y-axis data
    
    Returns
    -------
    [height, position, width] : ndarray
        Height, position and width of Lorentzian peak
    
    References
    ----------
    Routine is based on Caruana algorithm mention in paper by Hongwei Guo.
    'A Simple Algorithm for Fitting a Gaussian Function', IEEE Signal
    Processing Magazine, September 2011
    """
    # To avoid undefined 1/y
    maxy =np.max(y)
    ind = np.arange(len(y))
    mody = np.array(map(lambda i: maxy/100.0 if y[i] < maxy/100.0 else y[i], ind))             
    
    c, b, a = np.polyfit(x, 1/mody, 2)
    
    # Calculate fitted values of height, position and width
    height = (4*c) / (4*a*c - b**2)
    position = - b / (2*c)
    
    if (4*a*c - b**2) < 0:
        print >> sys.stdout, '\n Warning: Non-physical Fitting parameters. Try weighted or iterative least square methods.' 
        width = - np.sqrt(np.absolute(4*a - b**2/c)) / np.sqrt(c)
    else:
        width = np.sqrt(4*a - (b**2/c)) / np.sqrt(c)
    
    return np.array([height, position, width])   


## ---- Caruana's linear least square fitting ----  
def linalgfit(x, y):
    """
    Use Linear Algebra to estimate height, width and position of the
    Lorentzian peak.
    
    Parameters
    ----------
    x : ndarray
        x-axis data
        
    y : ndarray
        y-axis data
    
    Returns
    -------
    [height, position, width] : ndarray
        Height, position and width of Lorentzian peak
    
    References
    ----------
    Routine is based on Caruana algorithm mention in paper by Hongwei Guo.
    'A Simple Algorithm for Fitting a Gaussian Function', IEEE Signal
    Processing Magazine, September 2011
    """    
    # To avoid undefined 1/y
    maxy =np.max(y)
    ind = np.arange(len(y))
    mody = np.array(map(lambda i: maxy/100.0 if y[i] < maxy/100.0 else y[i], ind)) 
    
    # Common variables
    N = len(x)
    Sx = np.sum(x)
    Sx2 = np.sum(x**2)
    Sx3 = np.sum(x**3)
    Sx4 = np.sum(x**4)
    
    S1_y = np.sum(1/mody)
    Sx_y = np.sum(x/mody)
    Sx2_y = np.sum(x**2/mody)
                
    # Create a 3x3 matrix and a 3 dimensional vector
    X = np.array( [[N, Sx, Sx2], [Sx, Sx2, Sx3], [Sx2, Sx3, Sx4]] )
    Y = np.vstack(np.array( [S1_y, Sx_y, Sx2_y] ))
    
    # Solve linear equation using least square
    a, b, c = np.linalg.lstsq(X,Y)[0]
    
    # Calculate fitted values of height, position and width
    height = (4 * c) / (4*a*c - b**2)
    position = - b/(2 * c)

    if (4*a*c - b**2) < 0:
        print >> sys.stdout, '\n Warning: Non-physical Fitting parameters. Try weighted or iterative least square methods.' 
        width = - np.sqrt(np.absolute(4*a - b**2/c)) / np.sqrt(c)
    else:
        width = np.sqrt(4*a - (b**2/c)) / np.sqrt(c)

    return np.array( [height, position, width] )            
        
        
def wlstsq(x, y):
    """
    Weighted linear suqate fitting of Lorentzian peak. It estimates the position,
    height and width of the peak.
    
    Parameters
    ----------
    x : ndarray
        x-axis data
        
    y : ndarray
        y-axis data
    
    Returns
    -------
    [height, position, width] : ndarray
        Height, position and width of Lorentzian peak
    
    References
    ----------
    Routine is based on Caruana algorithm mention in paper by Hongwei Guo.
    'A Simple Algorithm for Fitting a Gaussian Function', IEEE Signal
    Processing Magazine, September 2011
    """ 
    # To avoid undefined 1/y
    maxy =np.max(y)
    ind = np.arange(len(y))
    mody = np.array(map(lambda i: maxy/100.0 if y[i] < maxy/100.0 else y[i], ind)) 
    
    # Common variables
    Sy4 = np.sum(mody**4)
    Sxy4 = np.sum(x * mody**4)
    Sx2y4 = np.sum(x**2 * mody**4)
    Sx3y4 = np.sum(x**3 * mody**4)
    Sx4y4 = np.sum(x**4 * mody**4)
    
    Sy3 = np.sum(mody**3)
    Sxy3 = np.sum(x * mody**3)
    Sx2y3 = np.sum(x**2 * mody**3)
    
    # Create a 3x3 matrix and a 3 dimensional vector
    X = np.array( [[Sy4, Sxy4, Sx2y4], [Sxy4, Sx2y4, Sx3y4], [Sx2y4, Sx3y4, Sx4y4]] )
    Y = np.vstack(np.array( [Sy3, Sxy3, Sx2y3] ))
    
    # Solve linear equation using least square
    a, b, c = np.linalg.lstsq(X,Y)[0]
    
    # Calculate fitted values of height, position and width
    height = (4 * c) / (4*a*c - b**2)
    position = - b/(2 * c)
    
    if (4*a*c - b**2) < 0:
        print >> sys.stdout, '\n Warning: Non-physical Fitting parameters. Try iterative weighted least square method.' 
        width = - np.sqrt(np.absolute(4*a - b**2/c)) / np.sqrt(c)
    else:
        width = np.sqrt(4*a - (b**2/c)) / np.sqrt(c)
        
    return np.array( [height, position, width] )      

                
def iterwlstsq(x, y, tol = 1e-6):
    """
    Iterative Weighted linear suqate fitting of Lorentzian peak. It estimates 
    the position, height and width of the peak.
    
    Parameters
    ----------
    x : ndarray
        x-axis data
        
    y : ndarray
        y-axis data
        
    tol : float
        convergence tolerance
    
    Returns
    -------
    [height, position, width] : ndarray
        Height, position and width of Lorentzian peak
    
    References
    ----------
    Routine is based on Caruana algorithm mention in paper by Hongwei Guo.
    'A Simple Algorithm for Fitting a Gaussian Function', IEEE Signal
    Processing Magazine, September 2011
    """     
    # To avoid undefined 1/y
    maxy =np.max(y)
    ind = np.arange(len(y))
    mody = np.array(map(lambda i: maxy/100.0 if y[i] < maxy/100.0 else y[i], ind)) 
    
    # Common variables
    aprev, bprev, cprev = 0.0, 0.0, 0.0
    niter = 0
    diffa, diffb, diffc = 1.0, 1.0, 1.0
    
    # Execute the loop till convergence criterion is fulfilled
    while (diffa > tol or diffb > tol or diffc > tol) and niter < 1000:
        print >> sys.stdout, '\n Iteration - ', niter+1
        
        if niter == 0:
            moditery = np.copy(mody)
        else:
            moditery = 1 / (aprev + bprev * x + cprev * x**2)    
            
        Sy4 = np.sum(moditery**4)
        Sxy4 = np.sum(x * moditery**4)
        Sx2y4 = np.sum(x**2 * moditery**4)
        Sx3y4 = np.sum(x**3 * moditery**4)
        Sx4y4 = np.sum(x**4 * moditery**4)
        
        Sy4_y = np.sum(moditery**4/mody)
        Sxy4_y = np.sum(x * moditery**4 / mody)
        Sx2y4_y = np.sum(x**2 * moditery**4 / mody)
        
        # Create a 3x3 matrix and a 3 dimensional vector
        X = np.array([[Sy4, Sxy4, Sx2y4], [Sxy4, Sx2y4, Sx3y4], [Sx2y4, Sx3y4, Sx4y4]])
        Y = np.vstack(np.array( [Sy4_y, Sxy4_y, Sx2y4_y] ))
    
        # Solve linear equation using least square
        a, b, c = np.linalg.lstsq(X,Y)[0]
        
        # Calculate fitted values of height, position and width
        height = (4 * c) / (4*a*c - b**2)
        position = - b/(2 * c)
        
        if (4*a*c - b**2) < 0:
            print >> sys.stdout, '\n Warning: Non-physical Fitting parameters.' 
            width = - np.sqrt(np.absolute(4*a - b**2/c)) / np.sqrt(c)
        else:
            width = np.sqrt(4*a - (b**2/c)) / np.sqrt(c)
            
        # calculate difference between previous and current value of parameters
        diffa, diffb, diffc = a - aprev, b - bprev, c - cprev
        aprev, bprev, cprev = a, b, c
        niter += 1

    if niter == 1000:
        print >> sys.stdout, 'Maximum iterations (1000) reached.'

    return np.array( [height, position, width] )                     