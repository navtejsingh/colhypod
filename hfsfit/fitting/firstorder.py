# firstorder.py - First order polynomial fitting

import sys
import numpy as np


def algsol(x, y):
    """
    Function to fit a straight line to data and estimate slope and 
    intercept of the line and corresponding errors using algebraic solution.

    Parameters
    ----------
    x : ndarray
        X-axis data
        
    y : ndarray
        Y-axis data
        
    Returns
    -------
    ndarray
        slope, intercept, SDslope, SDintercept
    
    Reference
    ---------    
    
    """
    # Number of input points
    N = x.shape[0]
    
    # Estimate intercept and slope of the straight line fitting            
    Sxx = np.sum((x - np.mean(x))**2)
    Syy = np.sum((y - np.mean(y))**2)
    Sxy = np.sum((x - np.mean(x)) * (y - np.mean(y)))
    calcSlope = Sxy / Sxx
    calcIntercept = np.mean(y) - calcSlope * np.mean(x)
    Sy = np.sqrt((Syy - Sxx * calcSlope**2) / (N - 2))
    SDSlope = Sy / np.sqrt(Sxx)
    SDIntercept = Sy * np.sqrt(1 / (N - (np.sum(x)**2 / np.sum(x**2))))
    
    return np.array([[calcSlope, calcIntercept], [SDSlope, SDIntercept]])


def polyFit(x, y):
    """
    Function to fit a straight line to data and estimate slope and 
    intercept of the line and corresponding errors using first order 
    polynomial fitting.

    Parameters
    ----------
    x : ndarray
        X-axis data
        
    y : ndarray
        Y-axis data
        
    Returns
    -------
    ndarray
        slope, intercept, SDslope, SDintercept
    
    Reference
    ---------    
    
    """
    # Number of input points
    N = x.shape[0]
                
    # Estimate slope and intercept of fitted line
    slope, intercept = np.polyfit(x, y, 1)
    
    # Calculate standard deviation of slope and intercept
    yhat = intercept + slope * x
    residual = y - yhat
    
    Sx2 = np.sum(x**2)
    Sxx = np.sum((x - np.mean(x))**2)
    Sy_x = np.sqrt(np.sum(residual**2) / (N -2))
    SDslope = Sy_x / np.sqrt(Sxx)
    SDintercept = Sy_x * np.sqrt(Sx2 / (N * Sxx))
    
    return np.array([[slope, intercept], [SDslope, SDintercept]])
                
                
def linalgFit(x, y):
    """
    Function to fit a straight line to data and estimate slope and 
    intercept of the line and corresponding errors using linear algebra.

    Parameters
    ----------
    x : ndarray
        X-axis data
        
    y : ndarray
        Y-axis data
        
    Returns
    -------
    ndarray
        slope, intercept, SDslope, SDintercept
    
    Reference
    ---------    
    
    """
    # Number of input points
    N = x.shape[0]
    
    Sx = np.sum(x)
    Sx2 = np.sum(x**2)
    Sy = np.sum(y)
    Sxy = np.sum(x * y)
    
    X = np.array([[N,Sx],[Sx,Sx2]])
    Y = np.array([[Sy], [Sxy]])
    
    intercept, slope = np.linalg.lstsq( X, Y )[0]
    
    # Calculate standard deviation of slope and intercept
    yhat = intercept + slope * x
    residual = y - yhat
    
    Sxx = np.sum((x - np.mean(x))**2)
    Sy_x = np.sqrt(np.sum(residual**2) / (N -2))
    SDslope = Sy_x / np.sqrt(Sxx)
    SDintercept = Sy_x * np.sqrt(Sx2 / (N * Sxx))
    
    return np.array([[slope, intercept], [SDslope, SDintercept]])


def MonteCarloWorker(indata):
    """
    Monte Carlo worker function for single processor and multicore processing
    
    Parameters
    ----------
    indata : ndarray
        input parameters for worker function
    
    Returns
    -------
    mcoeffs : tuple
        Fitted parameters
    """
    # Unpack input data
    ind, x, coeffs, SDnoise = indata
    
    print >> sys.stdout, '\n Monte Carlo iteration - ', ind + 1

    # Form polynomical equation based on number of coefficients
    lx = len(x)
    n_coeffs = len(coeffs)
    i, ymc = 0, np.zeros(lx)
    while i < n_coeffs:
        ymc += coeffs[i] * x**i
        i += 1
    
    # Add normal random noise to y value                          
    ymc += SDnoise * np.random.randn(lx)
        
    # Fit the noisy data using numpy polyfit    
    mcCoeffs = np.polyfit(x, ymc, n_coeffs-1)

    # return a tuple of slope and intercept
    return mcCoeffs
        

def MonteCarloSim(x, coeffs, SDnoise = 1.0, niter = 1000, parallel = False):
    """
    Monte Carlo Simulation routine for linear fitting using normal random 
    noise. Standard deviation of noise should be closer to the one calculated
    by algebraic method to get relatistic results.
    
    Parameters
    ----------
    x : ndarray
        X-axis data
        
    coeffs : ndarray
        Fitted coefficients
        
    SDnoise : float
        Standard deviation of random noise
        
    niter : int
        Number of Monte Carlo iterations. Default is 1000.
        
    parallel : bool
        Do parallel processing? Default is False.
    
    Returns
    -------
    MCResults : ndarray
        Monte Carlo result
    """
    # Populate input data list
    idata = []
    ind = range(niter)
    idata = map(lambda i: [i, x, coeffs, SDnoise], ind)
    
    # Import multiprocessing module
    try:
        import multiprocessing as mp
    except:
        n_cpus = 1
    else:
        n_cpus = mp.cpu_count()
    
    # Conditional loop for single and multicore processing
    if n_cpus == 1 or parallel == False:
        results = map(MonteCarloWorker, idata)
    else:
        pool = mp.Pool( n_cpus )
        results = pool.map(MonteCarloWorker, idata)

    # Convert result array to numpy num_coeffxniter array. The first
    # row is slope values and second row is intercept values
    MCResults = np.array(results).T

    # Output mean value of slope and intercept as well as 
    # standard deviation values
    return MCResults


def MonteCarlo(x, y, SDnoise = 1.0, niter = 1000, parallel = False):
    """
    Function to run Monte Carlo to determine mean slope and intercept. 
    Standard deviation of slope and intercept is also calculated.
    
    Parameters
    ----------
    x : ndarray
        X-axis data
        
    y : ndarray
        Y-axis data
        
    SDnoise : float
        standard deviation of noise
        
    niter : int
        number of Monte Carlo iterations
        
    parallel : bool
        run in parallel mode? Default if False
        
    
    Returns
    -------
    ndarray
        Mean slope and intercept and SD of slope and intercept
    
    References
    ----------
    
    """
    # Estimate slope and intercept of fitted line
    coeffs = np.polyfit(x, y, 1)
    
    # Input coefficient array is inverted for ease of processing
    results = MonteCarloSim(x, coeffs[-1::-1], SDnoise, niter, parallel)
    
    return np.array([[np.mean(results[0,:]), np.mean(results[1,:])], [np.std(results[0,:]), np.std(results[1,:])]])
            

def BootStrapWorker(indata):
    """
    Bootstrap worker function for single processor and multicore processing
    
    Parameters
    ----------
    indata : ndarray
        input parameters for worker function
    
    Returns
    -------
    mcCoeffs : tuple
        Fitted parameters
    """
    # Unpack input values
    i, x, y, coeffs, resample = indata

    print >> sys.stdout, '\n Bootstrap iteration - ', str(i+1)

    lx = len(x)
    n_coeffs = len(coeffs)
    resamplex, resampley = np.copy(x), np.copy(y)    
    if resample == 'case':
        # Case resampling - replacement of elements (duplicates allowed)
        j = 0
        while j < lx - 2:
            if np.random.rand() > 0.5:
                resamplex[j] = x[j+1]
                resampley[j] = y[j+1]
                j += 1

    elif resample == 'residual':
        # Calculate residuals
        i, yhat = 0, np.zeros(lx)
        while i < n_coeffs:
            yhat += coeffs[i] * x**i
            i += 1
        residual = y - yhat
        
        # Residual resampling by using residual of model and actual values.
        ind = np.arange(lx) 
        resampley = map(lambda i: yhat[i] + np.random.choice(residual), ind)

    elif resample == 'wild':
        # Calculate residuals
        i, yhat = 0, np.zeros(lx)
        while i < n_coeffs:
            yhat += coeffs[i] * x**i
            i += 1
        residual = y - yhat
        
        # Wild resampling by using residual of model and actual values
        # and multiplied by normal distributed noise
        ind = np.arange(lx) 
        resampley = map(lambda i: yhat[i] + np.random.choice(residual) * np.random.randn(), ind)
        
    else:
        print >> sys.stdout, 'Error: Only Case, Residual and Wild bootstrap resampling methods allowed. Exiting.' 
        sys.exit(-1)                             
    
    # Fit the noisy data using numpy polyfit    
    mcCoeffs = np.polyfit(resamplex, resampley, n_coeffs-1)

    # return a tuple of slope and intercept
    return mcCoeffs


def BootStrapSim(x, y, coeffs, resample = 'residual', numtrials = 1000, parallel = False):
    """
    Bootstrap Simulation routine for linear fitting.
    
    Parameters
    ----------
    x : ndarray
        X-axis data
    
    y: ndarray
        Y-axis data
                    
    coeffs : ndarray
        Fitted coefficients
        
    resample : string
        Resampling method. Default is residual.
        
    numtrials : int
        Number of bootstrap trials. Default is 1000.
        
    parallel : bool
        Do parallel processing? Default is False.
    
    Returns
    -------
    BSResults : ndarray
        Monte Carlo result
    """    
    # Populate input data list
    idata = []
    ind = range(numtrials)
    idata = map(lambda i: [i, x, y, coeffs, resample], ind)        

    # Import multiprocessing module
    try:
        import multiprocessing as mp
    except:
        n_cpus = 1
    else:
        n_cpus = mp.cpu_count()
                
    # Conditional loop for single and multicore processing
    if n_cpus == 1 or parallel == False:
        results = map(BootStrapWorker, idata)
    else:
        pool = mp.Pool( n_cpus )
        results = pool.map(BootStrapWorker, idata)

    # Create a numpy num_coeffs x numtrials array, with each row 
    # correspond to coeff. values in all the trials
    BSResults = np.array(results).T
    
    return BSResults


def BootStrap(x, y, resample = 'residual', numtrials = 1000, parallel = False):
    """
    Function to run Bootstrapping to determine mean slope and intercept. 
    Standard deviation of slope and intercept is also calculated.
    
    Parameters
    ----------
    x : ndarray
        X-axis data
        
    y : ndarray
        Y-axis data
        
    resample : string
        Resampling method. Default is residual.
        
    nimtrials : int
        Number of bootstrapping trials
        
    parallel : bool
        run in parallel mode? Default if False
        
    
    Returns
    -------
    ndarray
        Mean slope and intercept and SD of slope and intercept
    
    References
    ----------
    
    """
    # Estimate slope and intercept of fitted line
    coeffs = np.polyfit(x, y, 1)
    
    # Input coefficient array is inverted for ease of processing
    results = BootStrapSim(x, y, coeffs[-1::-1], resample, numtrials, parallel)
    
    return np.array([[np.mean(results[0,:]), np.mean(results[1,:]), np.std(results[0,:]), np.std(results[1,:])]])