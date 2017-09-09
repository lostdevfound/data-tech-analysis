import numpy as np
import sympy as sp
from math import *

def fourierCoef(vec, t, trigFunc):
    """
    Compute the coefficients for Fourier trig series
    vec -- 1D numpy array of data for Fourier series
    t -- integer that defines the frequency for a coefficient computation
    trigFunc -- string input "cos" or "sin"
    """
    a = 0
    x = sp.Symbol('x')

    if trigFunc != "sin" and trigFunc != "cos":
        raise ValueError("trigFunc argument must be 'sin' or 'cos'")

    # iterate through the vector and get vec[i] and vec[i+1]
    for i in range(len(vec) - 1):
        scaler = 2*pi / (len(vec) -1 )   # compute the scale value for compressing the domain into [0, 2pi]
        slope = (vec[i + 1] - vec[i]) / ((i + 1)*scaler - i*scaler)
        # put the data indexes into the domain of [0, 2pi]
        lowerBound = i*scaler
        upperBound = (i + 1)*scaler

        # compute the coefficients by taking an inner dot product. dot prod is defined as an integral of f(x) and trig functions
        if trigFunc == "cos":
            f = lambda x: (-1)*i*scaler*slope*sin(t*x)/t + vec[i]*sin(t*x)/t + slope*x*sin(t*x)/t + slope*cos(t*x)/t**2     # antiderivative

            if (t == 0):
                f = lambda x: slope*x**2/2 + x*(-i*scaler*slope + vec[i])
                a += (f(upperBound) - f(lowerBound)) * 1 / (2 * pi)
            else:
                a += (f(upperBound) - f(lowerBound)) * 1 / pi

        else:
            f = lambda x: i*scaler*slope*cos(t*x)/t - vec[i]*cos(t*x)/t - slope*x*cos(t*x)/t + slope*sin(t*x)/t**2      # antiderivative

            if (t == 0):
                a += 0
            else:
                a += (f(upperBound) - f(lowerBound)) * 1 / pi
    return a


def fourierTrigSeries(vec, n=10.0):
    """ Compute trigonometric Fourier series and return a tuple which contains
    a list of coefs and corresponding list of trig functions
    Keyword args:
    vec -- a 1D numpy array which contains the data points for Fourier series
    n -- number of terms in the series
    data -- python list which will be filled with decomposed frequencies
    """
    x = sp.Symbol('x')
    series = 0
    scaler = 2*sp.pi / (len(vec) -1 )
    trigTerms = []

    for t in range(n + 1):
        cosCoef = fourierCoef(vec, t, "cos")
        sinCoef = fourierCoef(vec, t, "sin")
        cos = sp.cos(t*x*scaler)
        sin = sp.sin(t*x*scaler)
        trigTerms.append((cosCoef, cos))    # list of tuples (coef, trigfunc(tx))
        trigTerms.append((sinCoef, sin))
        # Fourier series
        series += cosCoef*cos + sinCoef*sin

    return (sp.lambdify(x, series, modules=['numpy']), trigTerms)


def convolveFourierSeries(trigTerms, filterFactor, operation):
    """ Convolve the series in the frequency space using Gaussian function and return a new series and
    a list of tuples containing a coef value and the associated trig function
    Keyword args:
    trigTerms -- a list of tupes which contains a coef value and the associated trig term
    filterFactor -- a float which determines how much of convolution is applied on the curve, 0 value gives no convolution
    operation -- can be either 'highFreq' or 'lowFreq', if 'highFreq' is chosen then the method will filter high frequncies and vise-versa
    """
    if filterFactor < 0:
        raise ValueError('smoothingFactor should be greater or equal to zero')

    if operation != "highFreq" and operation != "lowFreq":
        raise ValueError("The third argument must be 'lowFreq' or 'highFreq'")

    convolveFunction = lambda x: e**((-1)*filterFactor*(x**2))   # Gaussian function for line smoothing

    if operation == "lowFreq" and filterFactor != 0:
        convolveFunction = lambda x: (-1)*e**((-1)*((1 - filterFactor)*x**2)/filterFactor) + 1

    x = sp.Symbol('x')
    coef, trigFuncs = zip(*trigTerms)

    coefList = np.array(coef)
    modList = np.linspace(0,1,len(coefList))     # generate the x axis number line
    modList = convolveFunction(modList)
    modCoef = np.multiply(coefList, modList)

    f = 0*x # define a 0 function
    newTrigTerms = []
    numLines = len(trigFuncs)

    for i in range(numLines):
        f += modCoef[i]* trigFuncs[i]
        newTrigTerms.append((modCoef, trigFuncs[i]))

    return (sp.lambdify(x, f,modules=['numpy'] ), newTrigTerms)


def ema(vec, i, a, data):
    """ Compute recursive exponential moving avearge for index i and store
    each ema value in 'data' vector.
    vec -- a 1D numpy array of data
    i -- integer index of a current datum
    a -- float smoothing value, a is between [0, 1]
    data -- 1D array of ema values
    """
    if (i == 0):
        data.append(vec[i])
        return vec[i]

    result = a*vec[i] + (1 - a)*ema(vec, i - 1, a, data)
    data.append(result)
    return result


def emaData(vec, a=0.5):
    """ A wraper for ema function. Compute ema for each datum and return the list of ema values
    vec -- a 1D numpy array of data
    a -- smoothing float value, a is between [0, 1]
    """
    data = []
    ema(vec, vec.size - 1, 0.5, data)
    return data
