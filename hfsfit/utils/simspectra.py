# simspectra.py - Simulated spectra with and without noise

## ---- Import python modules ----
import numpy as np
import peakshapes as _ps
import noisemodels as _nm


## ---- Global Variables ----
peakshapes = _ps.__all__
noisemodels = _nm.__all__


## ---- Model Peaks Function ----
def modelpeaks(xx, numpeaks, peakshape, positions, heights, widths, extra = 0.5):
    """
    Function to simulate ideal signal without noise for a specific peakshape.
    
    Parameters
    ----------
    xx : ndarray
        x-axis values
        
    numpeaks : int
        Number of peaks
        
    peakshape : string
        Peak shape
        
    positions : ndarray
        Position of the peaks. Should have length equal to numpeaks.
        
    heights : ndarray
        Height of peaks. Should have length equal to numpeaks.
        
    widths : ndarray
        Width of peaks. Should have length equal to numpeaks.
    
    extra : float
        Extra parameter for bigaussian, bilorentzian, GL, Voigt, and
        pearson peak shapes (see peakshapes.py for detail)
        
    Output
    ------
    model : ndarray
        Model 1D signal/spectra
    
    References
    ----------
    Implementation of MATLAB code from
    http://terpconnect.umd.edu/~toh/spectrum/functions.html    
    """
    numpeaks = int(numpeaks)
    
    # Number of peaks should be equal to length of positions and widths vector
    if len(positions) != numpeaks or len(widths) != numpeaks or len(heights) != numpeaks:
        raise ValueError('Error: Positions vector length is not same as number of peaks. Exiting.')
        
    # Check of input peakshape is valid
    if peakshape not in peakshapes:
        raise ValueError('Error: Input peak shape not supported at the moment. Try one of ', peakshapes, 'Exiting.')
        
    # Define a two-dimensional array for output signal
    yy = np.zeros( [numpeaks, xx.shape[0]] )
    
    # Generate multi-peak signal
    for i in range(0, numpeaks):
        if peakshape == 'gaussian':
            yy[i,:] = _ps.gaussian(positions[i], widths[i], xx)
        elif peakshape == 'lorentzian':
            yy[i,:] = _ps.lorentzian(positions[i], widths[i], xx)
        elif peakshape == 'bigaussian':
            yy[i,:] = _ps.bigaussian(positions[i], widths[i], xx, extra)
        elif peakshape == 'bilorentzian':
            yy[i,:] = _ps.bilorentzian(positions[i], widths[i], xx, extra)
        elif peakshape == 'voigt':
            yy[i,:] = _ps.voigt(positions[i], widths[i], xx, extra)
        elif peakshape == 'pearson':
            yy[i,:] = _ps.pearson(positions[i], widths[i], xx, extra)
        elif peakshape == 'gl':
            yy[i,:] = _ps.GL(positions[i], widths[i], xx, extra)
        elif peakshape == 'logistic':
            yy[i,:] = _ps.logistic(positions[i], widths[i], xx)
        elif peakshape == 'lognormal':
            yy[i,:] = _ps.lognormal(positions[i], widths[i], xx)
        else:
            pass
            
    # Multiply output signal with amplitude and add them to get final signal
    model = np.zeros(xx.shape[0])
    for i in range(0, numpeaks):
        model += heights[i] * yy[i,:]
    
    return model
    
    
## ---- Model Peaks with random noise ----
def noisymodelpeaks(xx, numpeaks, peakshape, positions, heights, widths, extra = 0.5, noisemodel = 'whitenoise'):
    """
    Function to simulate 1D noisy signal/spectra for a specific peakshape.
    
    Parameters
    ----------
    xx : ndarray
        x-axis values
        
    numpeaks : int
        Number of peaks
        
    peakshape : string
        Peak shape
        
    positions : ndarray
        Position of the peaks. Should have length equal to numpeaks.
        
    heights : ndarray
        Height of peaks. Should have length equal to numpeaks.
        
    widths : ndarray
        Width of peaks. Should have length equal to numpeaks.
    
    extra : float
        Extra parameter for bigaussian, bilorentzian, GL, Voigt, and
        pearson peak shapes (see peakshapes.py for detail)
        
    noisemodel : string
        Random noise model
        
    Output
    ------
    noisymodel : ndarray
        Noisy 1D signal/spectra
    
    References
    ----------
    Implementation of MATLAB code from
    http://terpconnect.umd.edu/~toh/spectrum/functions.html    
    """    
    if noisemodel not in noisemodels:
        raise ValueError('Error: Noise model not supported. Only ', noisemodels, ' are supported at the moment. Exiting.')
        
    model = modelpeaks(xx, numpeaks, peakshape, positions, heights, widths, extra)    
    
    noisymodel = np.zeros(xx.shape[0])
    
    if noisemodel == 'whitenoise':
        noisymodel = model + 1. * _nm.whitenoise(model)
    elif noisemodel == 'pinknoise':
        noisymodel = model + 1. * _nm.pinknoise(model)
    elif noisemodel == 'propnoise':
        noisymodel = model + 1. * _nm.propnoise(model)
    elif noisemodel == 'sqrtnoise':
        noisymodel = model + 1. * _nm.sqrtnoise(model)
    elif noisemodel == 'bimodalnoise':
        noisymodel = model + 1. * _nm.bimodalnoise(model)
    else:
        pass
        
    return noisymodel