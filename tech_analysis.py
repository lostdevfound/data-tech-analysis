import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
import seaborn as sns
from math import *

def coef(vec, t, trigFunc):
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
            f = lambda x: (-1)*i*scaler*slope*sin(t*x)/t + vec[i]*sin(t*x)/t + slope*x*sin(t*x)/t + slope*cos(t*x)/t**2
            # f = (-1)*i*scaler*slope*sp.sin(t*x)/t + vec[i]*sp.sin(t*x)/t + slope*x*sp.sin(t*x)/t + slope*sp.cos(t*x)/t**2

            if (t == 0):
                f = lambda x: slope*x**2/2 + x*(-i*scaler*slope + vec[i])
                # f = slope*x**2/2 + x*(-i*scaler*slope + vec[i])
                a += (f(upperBound) - f(lowerBound)) * 1 / (2 * sp.pi)
            else:
                a += (f(upperBound) - f(lowerBound)) * 1 / sp.pi

        else:
            f = lambda x: i*scaler*slope*cos(t*x)/t - vec[i]*cos(t*x)/t - slope*x*cos(t*x)/t + slope*sin(t*x)/t**2
            # f = i*scaler*slope*sp.cos(t*x)/t - vec[i]*sp.cos(t*x)/t - slope*x*sp.cos(t*x)/t + slope*sp.sin(t*x)/t**2

            if (t == 0):
                a += 0
            else:
                a += (f(upperBound) - f(lowerBound)) * 1 / sp.pi
    # print(trigFunc, "coef:", t, a)
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
        series += coef(vec, t, "cos")*sp.cos(t*x*scaler) + coef(vec, t, "sin")*sp.sin(t*x*scaler)

    return sp.lambdify(x, series, modules=['numpy'])


# test program
if __name__ == "__main__":
    xs = np.arange(0, 15, 0.1)
    v = np.random.rand(15)
    series = fourierTrigSeries(v, 6)
    plt.plot(xs, series(xs), '-b')
    plt.plot(v, 'gs--')
    plt.show()
