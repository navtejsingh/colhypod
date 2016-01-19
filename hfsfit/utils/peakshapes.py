# peakshapes.py - Various 1D Peak shapes


## ---- Import python modules ----
import numpy as np

__all__ = ['gaussian', 'bigaussian', 'expgaussian', 'lorentzian', 'bilorentzian', 'explorentzian', 'GL', 'logistic', 'lognormal', 'voigt', 'pearson']


## ---- Gaussian Signal ----
def gaussian(mu, wid, x):
    """
    Function to generate a 1D Gaussian signal centered at
    mu and wid wide (FWHM).
    
    Parameters
    ----------
    mu: float
        Central mean value of Gaussian peak
        
    wid: float
        FWHM of Gaussian peak. FWHM is related to sigma by the
        relation: FWHM = 2 * sqrt(2 * ln(2)) * sigma
        
    x: ndarray
        Input numpy array of numbers
        
    Output
    ------
    Numpy ndarray 
        Single Gaussian peak
      
    Reference
    ---------
    Implementation of MATLAB code from
    http://terpconnect.umd.edu/~toh/spectrum/functions.html#Peak_shape_functions
    """
    return np.exp(-((x - mu) / (0.6005612 * wid))**2)


## ---- BiGaussian Signal ----
def bigaussian(mu, wid, x, m = 0.5):
    """
    Function to generate a 1D BiGaussian signal centered at
    mu, wid wide (FWHM) and parameter m. Symmetrical BiGaussian
    is generated is m = 0.5
    
    Parameters
    ----------
    mu: float
        Central mean value of Gaussian peak
        
    wid: float
        FWHM of Gaussian peak. FWHM is related to sigma by the
        relation: FWHM = 2 * sqrt(2 * ln(2)) * sigma
        
    m: float
        Symmetry parameter. Default is 0.5
    
    x: ndarray
        Input numpy array of numbers
        
    Output
    ------
    Numpy ndarray 
        Single BiGaussian peak
      
    Reference
    ---------
    Implementation of MATLAB code from
    http://terpconnect.umd.edu/~toh/spectrum/functions.html#Peak_shape_functions
    """
    lx = x.shape[0]
    ix = np.where(x == mu)[0][0]
    
    y = np.ones(lx)
    y[0:ix] = gaussian(mu, wid * m, x[0:ix])
    y[ix+1:lx] = gaussian(mu, wid * (1 - m), x[ix+1:lx])        
    
    return y    


## ---- Exponetial broadened Gaussian ----
def expgaussian(mu, wid, timeconstant, x):
    """
    Function to generate a 1D Gaussian peak broadened by exponential signal.
    The signal is centered at mu, is wid wide (FWHM) and timeconstant t.
    
    Parameters
    ----------
    mu: float
        Central mean value of Gaussian peak
        
    wid: float
        FWHM of Gaussian peak. FWHM is related to sigma by the
        relation: FWHM = 2 * sqrt(2 * ln(2)) * sigma
        
    t: float
        Time constant for exponential function.
    
    x: ndarray
        Input numpy array of numbers
        
    Output
    ------
    Numpy ndarray 
        Single exponential boradened Gaussian peak
      
    Reference
    ---------
    Implementation of MATLAB code from
    http://terpconnect.umd.edu/~toh/spectrum/functions.html#Peak_shape_functions
    """    
    # Gaussian signal broadened by an exponetial signal
    g = gaussian(mu, wid, x)
    
    hly = np.round( len(g) / 2.0 )
    ey = np.r_[np.zeros(hly),g,np.zeros(hly)]
    fy = np.fft.fft(ey)
    a = np.exp(-(np.arange(len(fy))) / timeconstant )
    fa = np.fft.fft(a)
    fy1 = fy * fa
    ybz = np.real(np.fft.ifft(fy1)) / np.sum(a)
    yb = ybz[hly:len(ybz)-hly]
    
    return yb
    
                           
## ---- Lorentzian function ----
def lorentzian(mu, wid, x):
    """
    Function to generate a 1D Lorentzian peak. The peak is centered at mu and 
    is wid wide (FWHM).
    
    Parameters
    ----------
    mu: float
        Central mean value of Gaussian peak
        
    wid: float
        FWHM of Gaussian peak. FWHM is related to sigma by the
        relation: FWHM = 2 * sqrt(2 * ln(2)) * sigma
        
    x: ndarray
        Input numpy array of numbers
        
    Output
    ------
    Numpy ndarray 
        Single Lorentzian peak
      
    Reference
    ---------
    Implementation of MATLAB code from
    http://terpconnect.umd.edu/~toh/spectrum/functions.html#Peak_shape_functions
    """        
    return np.ones(len(x) ) / (1 + ( (x - mu) / (0.5 * wid) )**2)


## ---- BiLorentzian signal ----
def bilorentzian(mu, wid, x, m = 0.5):
    """
    Function to generate a 1D BiLorentzian peak. The peak is centered at mu,
    is wid wide (FWHM) and m symmetric.
    
    Parameters
    ----------
    mu: float
        Central mean value of Gaussian peak
        
    wid: float
        FWHM of Gaussian peak. FWHM is related to sigma by the
        relation: FWHM = 2 * sqrt(2 * ln(2)) * sigma
    
    m: float
        Symmetry parameter. Default is 0.5.
            
    x: ndarray
        Input numpy array of numbers
        
    Output
    ------
    Numpy ndarray 
        Single BiLorentzian peak
      
    Reference
    ---------
    Implementation of MATLAB code from
    http://terpconnect.umd.edu/~toh/spectrum/functions.html#Peak_shape_functions
    """       
    lx = x.shape[0]
    ix = np.where(x == mu)[0][0]
    
    y = np.ones(lx)
    y[0:ix] = lorentzian( mu, wid * m, x[0:ix] )
    y[ix+1:lx] = lorentzian( mu, wid * (1 - m), x[ix+1:lx] )  
    
    return y


## ---- Exponential broadened Lorentzian ----
def explorentzian(mu, wid, timeconstant, x):
    """
    Function to generate a 1D exponential broadened BiLorentzian peak. The peak 
    is centered at mu, is wid wide (FWHM) and exponential function time constant
    t.
    
    Parameters
    ----------
    mu: float
        Central mean value of Gaussian peak
        
    wid: float
        FWHM of Gaussian peak. FWHM is related to sigma by the
        relation: FWHM = 2 * sqrt(2 * ln(2)) * sigma
    
    t: float
        Time constant.
            
    x: ndarray
        Input numpy array of numbers
        
    Output
    ------
    Numpy ndarray 
        Single exponential broadened Lorentzian peak
      
    Reference
    ---------
    Implementation of MATLAB code from
    http://terpconnect.umd.edu/~toh/spectrum/functions.html#Peak_shape_functions
    """           
    g = lorentzian( mu, wid, x )
    
    hly = np.round( len(g) / 2.0 )
    ey = np.r_[np.zeros(hly),g,np.zeros(hly)]
    fy = np.fft.fft(ey)
    a = np.exp(-(np.arange(len(fy))) / timeconstant )
    fa = np.fft.fft(a)
    fy1 = fy * fa
    ybz = np.real(np.fft.ifft(fy1)) / np.sum(a)
    yb = ybz[hly:len(ybz)-hly]
    
    return yb


## ---- Gaussian/Lorentzian blend ----
def GL(mu, wid, x, m = 0.5):
    """
    Function to generate a 1D Gaussian-Lorentzian peak. The peak 
    is centered at pos, is wid wide (FWHM) and with blending parameter m.
    
    Parameters
    ----------
    mu: float
        Peak center
        
    wid: float
        FWHM of Gaussian peak. FWHM is related to sigma by the
        relation: FWHM = 2 * sqrt(2 * ln(2)) * sigma
    
    m: float
        Blending constant. Default is 0.5.
            
    x: ndarray
        Input numpy array of numbers
        
    Output
    ------
    Numpy ndarray 
        Single blended Gaussian-Lorentzian peak.
      
    Reference
    ---------
    Implementation of MATLAB code from
    http://terpconnect.umd.edu/~toh/spectrum/functions.html#Peak_shape_functions
    """     
    return m * gaussian(mu, wid, x) + (1 - m) * lorentzian(mu, wid, x) 


## ---- Logistic Function ----
def logistic(mu, hw, x):
    """
    Function to generate a logistic peak, centered at mu with half width
    half maximum hw.
    
    Parameters
    ----------
    mu: float
        Peak center
        
    hw: float
        HWHM of logistic peak. HWHM is related to sigma by the
        relation: FWHM = sqrt(2 * ln(2)) * sigma
    
    x: ndarray
        Input numpy array of numbers
        
    Output
    ------
    Numpy ndarray 
        Single logistic peak.
      
    Reference
    ---------
    Implementation of MATLAB code from
    http://terpconnect.umd.edu/~toh/spectrum/functions.html#Peak_shape_functions
    """        
    n = np.exp(- ((x-mu)/(.477*hw))**2)
    return (2. * n)/( 1 + n)


## ---- Lognormal function ----
def lognormal(mu, hw, x):
    """
    Function to generate a lognormal peak, centered at mu with half width
    half maximum hw.
    
    Parameters
    ----------
    mu: float
        Peak center
        
    hw: float
        HWHM of logistic peak. HWHM is related to sigma by the
        relation: FWHM = sqrt(2 * ln(2)) * sigma
    
    x: ndarray
        Input numpy array of numbers
        
    Output
    ------
    Numpy ndarray 
        Single lognormal peak.
      
    Reference
    ---------
    Implementation of MATLAB code from
    http://terpconnect.umd.edu/~toh/spectrum/functions.html#Peak_shape_functions
    """    
    return np.exp(-( np.log(x/mu) / (0.01*hw) )**2)


## ---- Voigt Function ----
def voigt(pos, gD, xx, alpha = 0.5):
    """
    Function to generate a lognormal peak, centered at pos with Voigt width
    gD.
    
    Parameters
    ----------
    pos: float
        Peak center
        
    gd: float
        Voigt width
    
    alpha: float
        ratio of Lorentzian width (gL) and Voigt width (gD)
        
    xx: ndarray
        Input numpy array of numbers
        
    Output
    ------
    Numpy ndarray 
        Single Voigt peak.
      
    Reference
    ---------
    Implementation of MATLAB code from
    http://terpconnect.umd.edu/~toh/spectrum/functions.html#Peak_shape_functions
    """     
    gL = alpha * gD
    gV = 0.5346 * gL + np.sqrt(0.2166 * gL**2 + gD**2)
    x = gL/gV
    y = np.abs(xx-pos) / gV
    g = 1/(2*gV*(1.065 + 0.447*x + 0.058*x**2))*((1-x)*np.exp(-0.693*y**2) + (x/(1+y**2)) + 0.016*(1-x)*x*(np.exp(-0.0841*y**2.25)-1./(1 + 0.021*y**2.25)));
    g = g / np.max(g)
    
    return g
    

## ---- Pearson Function ----     
def pearson(pos, wid, x, m = 1):
    """
    Function to generate a Pearson peak, centered at pos with width wid and 
    shape number m.
    
    Parameters
    ----------
    pos: float
        Peak center
        
    wid: float
        Peak width
    
    m: float
        Shape number. Default is 1.
        m = 1: Lorentzian
          > 20: Gaussian
          < 1: Cusp  
        
    x: ndarray
        Input numpy array of numbers
        
    Output
    ------
    Numpy ndarray 
        Single Voigt peak.
      
    Reference
    ---------
    Implementation of MATLAB code from
    http://terpconnect.umd.edu/~toh/spectrum/functions.html#Peak_shape_functions
    """     
    return np.ones(len(x)) / (1+(( x-pos) / ((0.5**(2/m)) * 4.62 * wid ))**2)**m