# nlfit.py - non-linear fitting using function minimization


import numpy as np
from hfsfit.utils.peakshapes import *
#from utils.peakshapes import *


def fitGaussian(parameters, x, y, npeaks):
    """
    Minimization funtion to fit Gaussian peaks in a spectra. Parameters are
    in a python list or a numpy vector with 3 first guess values for each 
    peak - height, position and width. Length of parameters list/array should
    be 3 * npeaks (number of peaks)
    
    Parameters
    ----------
    parameters : ndarray or list
        Peak guess parameters (height, position, width).
        
    x : ndarray
        X-axis data
        
    y : ndarray
        Y-axis data
        
    npeaks : int
        Number of peaks in spectra
    
    Returns
    -------
    SSE : float
        Sum of squares of residuals
    """
    # Create a numpy array of same length as signal        
    yhat = np.zeros(len(x))
    
    # Iterate through all the peaks to create modelled signal
    for i in range(npeaks):
        yhat += parameters[3*i] * gaussian( parameters[3*i+1], parameters[3*i+2], x )
    
    # Minimize SSE i.e. sum of square of residuals
    return np.sqrt(np.sum((yhat - y)**2))/(2*len(x))
    
    
def fitLorentzian(parameters, x, y, npeaks):
    """
    Minimization funtion to fit Lorentzian peaks in a spectra. Parameters are
    in a python list or a numpy vector with 3 first guess values for each 
    peak - height, position and width. Length of parameters list/array should
    be 3 * npeaks (number of peaks)
    
    Parameters
    ----------
    parameters : ndarray or list
        Peak guess parameters (height, position, width).
        
    x : ndarray
        X-axis data
        
    y : ndarray
        Y-axis data
        
    npeaks : int
        Number of peaks in spectra
    
    Returns
    -------
    SSE : float
        Sum of squares of residuals
    """
    # Create a numpy array of same length as signal        
    yhat = np.zeros(len(x))
    
    # Iterate through all the peaks to create modelled signal
    for i in range(npeaks):
        yhat += parameters[3*i] * lorentzian( parameters[3*i+1], parameters[3*i+2], x )
    
    # Minimize SSE i.e. sum of square of residuals
    return np.sqrt(np.sum((yhat - y)**2))/(2*len(x))    



def fitLogistic(parameters, x, y, npeaks):
    """
    Minimization funtion to fit Logistic peaks in a spectra. Parameters are
    in a python list or a numpy vector with 3 first guess values for each 
    peak - height, position and width. Length of parameters list/array should
    be 3 * npeaks (number of peaks)
    
    Parameters
    ----------
    parameters : ndarray or list
        Peak guess parameters (height, position, width).
        
    x : ndarray
        X-axis data
        
    y : ndarray
        Y-axis data
        
    npeaks : int
        Number of peaks in spectra
    
    Returns
    -------
    SSE : float
        Sum of squares of residuals
    """
    # Create a numpy array of same length as signal        
    yhat = np.zeros(len(x))
    
    # Iterate through all the peaks to create modelled signal
    for i in range( npeaks ):
        yhat += parameters[3*i] * logistic( parameters[3*i+1], parameters[3*i+2], x )
    
    # Minimize SSE i.e. sum of square of residuals
    return np.sqrt(np.sum((yhat - y)**2))/(2*len(x))   
    
    
def fitLognormal(parameters, x, y, npeaks):
    """
    Minimization funtion to fit Lognormal peaks in a spectra. Parameters are
    in a python list or a numpy vector with 3 first guess values for each 
    peak - height, position and width. Length of parameters list/array should
    be 3 * npeaks (number of peaks)
    
    Parameters
    ----------
    parameters : ndarray or list
        Peak guess parameters (height, position, width).
        
    x : ndarray
        X-axis data
        
    y : ndarray
        Y-axis data
        
    npeaks : int
        Number of peaks in spectra
    
    Returns
    -------
    SSE : float
        Sum of squares of residuals
    """
    # Create a numpy array of same length as signal        
    yhat = np.zeros(len(x))
    
    # Iterate through all the peaks to create modelled signal
    for i in range( npeaks ):
        yhat += parameters[3*i] * lognormal( parameters[3*i+1], parameters[3*i+2], x )
    
    # Minimize SSE i.e. sum of square of residuals
    return np.sqrt(np.sum((yhat - y)**2))/(2*len(x))     
    

def fitPearson(parameters, x, y, shapeconstant, npeaks):
    """
    Minimization funtion to fit Pearson peaks in a spectra. Parameters are
    in a python list or a numpy vector with 3 first guess values for each 
    peak - height, position and width. Length of parameters list/array should
    be 3 * npeaks (number of peaks)
    
    Parameters
    ----------
    parameters : ndarray or list
        Peak guess parameters (height, position, width).
        
    x : ndarray
        X-axis data
        
    y : ndarray
        Y-axis data

    shapeconstant : float
        Shape constant for Pearson peaks    
                        
    npeaks : int
        Number of peaks in spectra
    
    Returns
    -------
    SSE : float
        Sum of squares of residuals
    """
    # Create a numpy array of same length as signal        
    yhat = np.zeros(len(x))
    
    # Iterate through all the peaks to create modelled signal
    for i in range( npeaks ):
        yhat += parameters[3*i] * pearson( parameters[3*i+1], parameters[3*i+2], x, shapeconstant )
    
    # Minimize SSE i.e. sum of square of residuals
    return np.sqrt(np.sum((yhat - y)**2))/(2*len(x)) 



def fitVoigt(parameters, x, y, shapeconstant, npeaks):
    """
    Minimization funtion to fit Voigt peaks in a spectra. Parameters are
    in a python list or a numpy vector with 3 first guess values for each 
    peak - height, position and width. Length of parameters list/array should
    be 3 * npeaks (number of peaks)
    
    Parameters
    ----------
    parameters : ndarray or list
        Peak guess parameters (height, position, width).
        
    x : ndarray
        X-axis data
        
    y : ndarray
        Y-axis data

    shapeconstant : float
        Shape constant for Voigt peaks    
                        
    npeaks : int
        Number of peaks in spectra
    
    Returns
    -------
    SSE : float
        Sum of squares of residuals
    """
    # Create a numpy array of same length as signal        
    yhat = np.zeros(len(x))
    
    # Iterate through all the peaks to create modelled signal
    for i in range( npeaks ):
        yhat += parameters[3*i] * voigt( parameters[3*i+1], parameters[3*i+2], x, shapeconstant )
    
    # Minimize SSE i.e. sum of square of residuals
    return np.sqrt(np.sum((yhat - y)**2))/(2*len(x)) 
    

    
def fitGL(parameters, x, y, shapeconstant, npeaks):
    """
    Minimization funtion to fit GL peaks in a spectra. Parameters are
    in a python list or a numpy vector with 3 first guess values for each 
    peak - height, position and width. Length of parameters list/array should
    be 3 * npeaks (number of peaks)
    
    Parameters
    ----------
    parameters : ndarray or list
        Peak guess parameters (height, position, width).
        
    x : ndarray
        X-axis data
        
    y : ndarray
        Y-axis data

    shapeconstant : float
        Shape constant for GL peaks    
                        
    npeaks : int
        Number of peaks in spectra
    
    Returns
    -------
    SSE : float
        Sum of squares of residuals
    """
    # Create a numpy array of same length as signal        
    yhat = np.zeros(len(x))
    
    # Iterate through all the peaks to create modelled signal
    for i in range( npeaks ):
        yhat += parameters[3*i] * GL( parameters[3*i+1], parameters[3*i+2], x, shapeconstant )
    
    # Minimize SSE i.e. sum of square of residuals
    return np.sqrt(np.sum((yhat - y)**2))/(2*len(x))         


    

def fitExpGaussian(parameters, x, y, timeconstant, npeaks):
    """
    Minimization funtion to fit exponential Gaussian peaks in a spectra. Parameters
    are in a python list or a numpy vector with 3 first guess values for each 
    peak - height, position and width. Length of parameters list/array should
    be 3 * npeaks (number of peaks)
    
    Parameters
    ----------
    parameters : ndarray or list
        Peak guess parameters (height, position, width).
        
    x : ndarray
        X-axis data
        
    y : ndarray
        Y-axis data

    timeconstant : float
        Time constant for exponential Gaussian peak
                        
    npeaks : int
        Number of peaks in spectra
    
    Returns
    -------
    SSE : float
        Sum of squares of residuals
    """
    # Create a numpy array of same length as signal        
    yhat = np.zeros(len(x))
    
    # Iterate through all the peaks to create modelled signal
    for i in range( npeaks ):
        yhat += parameters[3*i] * expgaussian( parameters[3*i+1], parameters[3*i+2], x, timeconstant )
    
    # Minimize SSE i.e. sum of square of residuals
    return np.sqrt(np.sum((yhat - y)**2))/(2*len(x))         

                
   
def fitExpLorentzian(parameters, x, y, timeconstant, npeaks):
    """
    Minimization funtion to fit exponential Lorentzian peaks in a spectra. Parameters
    are in a python list or a numpy vector with 3 first guess values for each 
    peak - height, position and width. Length of parameters list/array should
    be 3 * npeaks (number of peaks)
    
    Parameters
    ----------
    parameters : ndarray or list
        Peak guess parameters (height, position, width).
        
    x : ndarray
        X-axis data
        
    y : ndarray
        Y-axis data

    timeconstant : float
        Time constant for exponential Lorentzian peak
                        
    npeaks : int
        Number of peaks in spectra
    
    Returns
    -------
    SSE : float
        Sum of squares of residuals
    """
    # Create a numpy array of same length as signal        
    yhat = np.zeros(len(x))
    
    # Iterate through all the peaks to create modelled signal
    for i in range( npeaks ):
        yhat += parameters[3*i] * explorentzian( parameters[3*i+1], parameters[3*i+2], x, timeconstant )
    
    # Minimize SSE i.e. sum of square of residuals
    return np.sqrt(np.sum((yhat - y)**2))/(2*len(x))                 
    

 
def fitBiGaussian(parameters, x, y, shapeconstant, npeaks):
    """
    Minimization funtion to fit BiGaussian peaks in a spectra. Parameters are
    in a python list or a numpy vector with 3 first guess values for each 
    peak - height, position and width. Length of parameters list/array should
    be 3 * npeaks (number of peaks)
    
    Parameters
    ----------
    parameters : ndarray or list
        Peak guess parameters (height, position, width).
        
    x : ndarray
        X-axis data
        
    y : ndarray
        Y-axis data

    shapeconstant : float
        Shape constant for bilorentzian        
    
    npeaks : int
        Number of peaks in spectra
    
    Returns
    -------
    SSE : float
        Sum of squares of residuals
    """
    # Create a numpy array of same length as signal        
    yhat = np.zeros(len(x))
    
    # Iterate through all the peaks to create modelled signal
    for i in range( npeaks ):
        yhat += parameters[3*i] * bigaussian( parameters[3*i+1], parameters[3*i+2], x, shapeconstant )
    
    # Minimize SSE i.e. sum of square of residuals
    return np.sqrt(np.sum((yhat - y)**2))/(2*len(x))     
    
        

def fitBiLorentzian(parameters, x, y, shapeconstant, npeaks):
    """
    Minimization funtion to fit BiLorentzian peaks in a spectra. Parameters are
    in a python list or a numpy vector with 3 first guess values for each 
    peak - height, position and width. Length of parameters list/array should
    be 3 * npeaks (number of peaks)
    
    Parameters
    ----------
    parameters : ndarray or list
        Peak guess parameters (height, position, width).
        
    x : ndarray
        X-axis data
        
    y : ndarray
        Y-axis data
    
    shapeconstant : float
        Shape constant for bilorentzian
                
    npeaks : int
        Number of peaks in spectra
    
    Returns
    -------
    SSE : float
        Sum of squares of residuals
    """
    # Create a numpy array of same length as signal        
    yhat = np.zeros(len(x))
    
    # Iterate through all the peaks to create modelled signal
    for i in range( npeaks ):
        yhat += parameters[3*i] * bilorentzian( parameters[3*i+1], parameters[3*i+2], x, shapeconstant )
    
    # Minimize SSE i.e. sum of square of residuals
    return np.sqrt(np.sum((yhat - y)**2))/(2*len(x))                   