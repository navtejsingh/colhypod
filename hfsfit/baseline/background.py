# background.py - determine signal/spectra background


## ---- Import Python Modules ----
import numpy as np


def linear(xx, yy):
    """
    Function to fit linear baseline (background) to input 1D signal/spectra
    
    Parameters
    ----------
    xx : ndarray
        X-axis values in numpy array
        
    yy : ndarray
        Y-axis values in numpy array 
    
    Output
    ------
    bkg: ndarray
        Linear background or baseline
    
    Reference
    ---------
    Implementation of MATLAB code from
    http://terpconnect.umd.edu/~toh/spectrum/findpeaksb.m
    """
    xx = np.asarray(xx)
    if len(xx.shape) != 1:
        raise ValueError('Only 1D signal spectra can be processed.')

    ssize = xx.shape[0]
    bkgsize = round(ssize/10)
    if bkgsize < 2:
        bkgsize = 2
     
    XX1 = xx[0:round(ssize/bkgsize)+1]
    XX2 = xx[(ssize-round(ssize/bkgsize)):ssize+1]
    Y1 = yy[0:(round(ssize/bkgsize))+1]
    Y2 = yy[(ssize-round(ssize/bkgsize)):ssize+1]
    bkgcoef = np.polyfit(np.concatenate((XX1,XX2)), np.concatenate((Y1,Y2)),1)  # Fit straight line to sub-group of points
    bkg = np.polyval(bkgcoef, xx)
    
    return bkg


def quadratic(xx, yy):
    """
    Function to fit linear baseline (background) to input 1D signal/spectra
    
    Parameters
    ----------
    xx : ndarray
        X-axis values in numpy array
        
    yy : ndarray
        Y-axis values in numpy array 
    
    Output
    ------
    bkg: ndarray
        Quadratic background or baseline
    
    Reference
    ---------
    Implementation of MATLAB code from
    http://terpconnect.umd.edu/~toh/spectrum/findpeaksb.m
    """
    xx = np.asarray(xx)
    if len(xx.shape) != 1:
        raise ValueError('Only 1D signal.spectra can be processed.')

    ssize = xx.shape[0]
    bkgsize = round(ssize/10)
    if bkgsize < 2:
        bkgsize = 2
     
    XX1 = xx[0:round(ssize/bkgsize)+1]
    XX2 = xx[(ssize-round(ssize/bkgsize)):ssize+1]
    Y1 = yy[0:round(ssize/bkgsize)+1]
    Y2 = yy[(ssize-round(ssize/bkgsize)):ssize+1]
    bkgcoef = np.polyfit(np.concatenate((XX1,XX2)), np.concatenate((Y1,Y2)), 2)  # Fit parabola to sub-group of points
    bkg = np.polyval(bkgcoef, xx)
    
    return bkg