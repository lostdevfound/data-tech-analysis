import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
import seaborn as sns
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
        scaler = 2*sp.pi / (len(vec) -1 )   # compute the scale value for compressing the domain into [0, 2pi]
        slope = (vec[i + 1] - vec[i]) / ((i + 1)*scaler - i*scaler)
        # put the data indexes into the domain of [0, 2pi]
        lowerBound = i*scaler
        upperBound = (i + 1)*scaler

        # compute the series of integrals with bounds i to i + 1
        if trigFunc == "cos":
            f = lambda x: (-1)*i*scaler*slope*sin(t*x)/t + vec[i]*sin(t*x)/t + slope*x*sin(t*x)/t + slope*cos(t*x)/t**2     # antiderivative

            if (t == 0):
                f = lambda x: slope*x**2/2 + x*(-i*scaler*slope + vec[i])
                a += (f(upperBound) - f(lowerBound)) * 1 / (2 * sp.pi)
            else:
                a += (f(upperBound) - f(lowerBound)) * 1 / sp.pi

        else:
            f = lambda x: i*scaler*slope*cos(t*x)/t - vec[i]*cos(t*x)/t - slope*x*cos(t*x)/t + slope*sin(t*x)/t**2      # antiderivative

            if (t == 0):
                a += 0
            else:
                a += (f(upperBound) - f(lowerBound)) * 1 / sp.pi
    # print(trigFunc, "fourierCoef:", t, a)
    return a


def fourierTrigSeries(vec, n=10):
    """ Compute trigonometric Fourier series.
    Keyword args:
    vec -- a 1D numpy array which contains the data points for Fourier series
    n -- number of terms in the series
    """
    x = sp.Symbol('x')
    series = 0
    scaler = 2*sp.pi / (len(vec) -1 )

    for t in range(n):
        series += fourierCoef(vec, t, "cos")*sp.cos(t*x*scaler) + fourierCoef(vec, t, "sin")*sp.sin(t*x*scaler)

    return sp.lambdify(x, series, modules=['numpy'])

def ema(vec, i, a, data):
    """ Compute recursive exponential moving avearge for index i and store
    each ema value in data vector.
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
    """ A wraper for ema function. Compute ema vector
    vec -- a 1D numpy array of data
    a -- smoothing float value, a is between [0, 1]
    """
    data = []
    ema(vec, vec.size - 1, 0.5, data)
    return data



# test program
if __name__ == "__main__":
    xs = np.arange(0, 15, 0.1)
    v = np.random.rand(15)
    # series = fourierTrigSeries(v, 8)
    # plt.plot(xs, series(xs), '-b', label='Fourier Trig Series, n=8')
    emaVector = emaData(v, 0.5)
    plt.plot(v, 'gs--', label='Original Data')
    plt.plot(emaVector, 'b', label='EMA')
    plt.title('Fourier Series')
    plt.show()
