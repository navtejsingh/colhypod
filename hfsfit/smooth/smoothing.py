# smoothing.py - Signal smoothing routines

## ---- Import python modules ----
import numpy as np
from numpy.linalg import norm
from scipy.fftpack import dct,idct
from scipy.optimize import fmin_l_bfgs_b

from math import factorial


## ---- Windowed smoothing function ----
def winsmooth(signal, window_len = 11, window = 1):
    """
    Function to smooth input 1D signal.spectra
    
    Parameters
    ----------
    signal : ndarray
        Input 1D signal/spectra
        
    window_len : int
        Smoothing window length
        
    window : int
        smoothing window type. Default is 1 ('flat'). Options are 
        1 : flat 
        2 : hanning 
        3 : hamming 
        4 : bartlett 
        5 : blackman
    
    
    Output
    ------
    ssignal : ndarray
        Smoothered signal/spectra
    
    
    Reference
    ---------
    Code taken from http://wiki.scipy.org/Cookbook/SignalSmooth
    """
    window_functions = {1: 'flat', 2: 'hanning', 3: 'hamming', 4: 'bartlett', 5: 'blackman'}
    
    if signal.ndim != 1:
        raise TypeError('Error: Only 1D signals are processed. Exiting.')
        
    if signal.size < window_len:
        raise TypeError('Error: Input signal need to be bigger than window size. Exiting.')
        
    if not window in window_functions.keys():
        raise ValueError("Only window types " + window_functions.values() + " allowed.")
              
    s = np.r_[signal[window_len-1:0:-1],signal,signal[-1:-window_len:-1]]
    
    if window == 1:
        w = np.ones(window_len, 'd')
    else:
        w = eval( 'np.' + window_functions[window] + '(window_len)' )

    ssignal = np.convolve( w/w.sum(), s, mode = 'valid' )
    
    return ssignal[((window_len - 1)/2):-(window_len/2)]

    
## ---- Savitzky Golay Filter ----    
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """
    Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.

    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
        
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
        
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()

    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
        
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
        
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
        
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))

    return np.convolve( m[::-1], y, mode='valid' )

    
def GCVscore(p,n,lamb,dcty):
    s = 10**p
    gamma = 1 / (1+(s*np.absolute(lamb))**2)
    rss = norm(dcty * (gamma - 1))**2
    trh = np.sum(gamma)
    gcvs = rss/n/(1-trh/float(n))**2

    return gcvs
                

def robustsmooth(signal, ns0=10):
    """
    Function to smooth 1D signal using robust smoothing method. It is based
    on algorithm in paper 'Robust smoothing of gridded data in one and higher 
    dimensions with missing values' by Damien Garcia
    
    Parameters
    ----------
    signal : ndarray
        Input 1D signal/spectra
        
    ns0 : int
        initial guess value
    
    Returns
    -------
    smooth_signal : ndarray
        Smoothered signal/spectra
    
    Reference
    ---------
    
    """
    # Determine length and shape of signal
    N = len(signal)
    Signalsize = signal.shape
    
    # Calculate bounding limit for s smoothing parameter
    Nsmooth = sum(np.array(Signalsize) != 1)
    hmin, hmax = 1e-6, 0.99
    sMinBnd = np.sqrt((((1+np.sqrt(1+8*hmax**(2./Nsmooth)))/4./hmax**(2./Nsmooth))**2-1)/16.)
    sMaxBnd = np.sqrt((((1+np.sqrt(1+8*hmin**(2./Nsmooth)))/4./hmin**(2./Nsmooth))**2-1)/16.)

    # Determine tensor Lambda
    idx = np.linspace(1,N,N)
    Lambda = -2 + 2*np.cos((idx - 1)*np.pi/N)            
    
    # Determine discrete cosine transformation of signal
    dcty = dct(signal,type=2,norm='ortho')            

    # Determine initial guess value of smoothing parameter s        
    ss = np.arange(ns0)*(1./(ns0-1.))*(np.log10(sMaxBnd)-np.log10(sMinBnd))+ np.log10(sMinBnd)
    g = np.zeros(len(ss))
    for i, p in enumerate(ss):
        g[i] = GCVscore(p,N,Lambda,dcty)
    
    s0 = np.absolute([ss[g == g.min()]])
    print 's0: ', s0
    
    # Minimize smoothing parameters s
    res = fmin_l_bfgs_b(GCVscore, s0, fprime = None, factr = 10., args=(N,Lambda,dcty), approx_grad=True, bounds = [(np.log10(sMinBnd),np.log10(sMaxBnd))])

    # Construct smoothered signal
    gamma = 1 / (1+(10**res[0]*Lambda)**2)
    smooth_signal = idct(gamma*dcty,type=2,norm='ortho')
    
    return smooth_signal  
    

            
            
    