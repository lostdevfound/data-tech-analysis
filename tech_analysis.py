import numpy as np
import sympy as sp
# from matplotlib import pyplot as plt
# import seaborn as sns
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
    # print(trigFunc, "fourierCoef:", t, a)
    return a


def fourierTrigSeries(vec, n=10):
    """ Compute trigonometric Fourier series.
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
        trigTerms.append((cosCoef, cos))    # list of tuples (coef, cos(tx))
        trigTerms.append((sinCoef, sin))
        # Fourier series
        series += cosCoef*cos + sinCoef*sin

    return (sp.lambdify(x, series, modules=['numpy']), trigTerms)

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



# test program
if __name__ == "__main__":
    # sns.set_style("whitegrid")
    #
    #
    # xs = np.arange(0, 15, 0.1)
    # v = np.random.rand(15)
    # plt.figure(1)
    # plt.subplot(311)
    # series, trigTerms = fourierTrigSeries(v, 8)
    # plt.title('Fourier Analysis')
    # plt.plot(v, 'gs--', label='Original Data')
    # plt.plot(xs, series(xs), '-b', label='Fourier Trig Series, n=8')
    # plt.legend(loc='lower left')
    #
    # plt.subplot(323)
    # coef,trigfuncs = zip(*trigTerms)
    # markerline, stemlines, baseline = plt.stem(coef)
    # plt.xticks([x for x in range(len(coef))], trigfuncs, rotation=15)
    # plt.tick_params(axis='both', which='major', labelsize=8)
    # plt.title('frequency amp')
    #
    # plt.subplot(324)
    # x = sp.Symbol('x')
    # plt.title('frequency decomposition')
    # for i in range(len(trigfuncs)):
    #     if i < 2:
    #         continue
    #     f = sp.lambdify(x, trigfuncs[i], modules=['numpy'])
    #     line, = plt.plot(xs, coef[i]*f(xs))
    #     line.set_linewidth(0.8)
    #
    # emaVector = emaData(v, 0.5)
    # plt.subplot(313)
    # plt.title('Exponential Moving Average')
    # plt.plot(v, 'gs--', label='Original Data')
    # plt.plot(emaVector, 'k', label='EMA')
    # plt.legend(loc='lower left')
    # plt.tight_layout(pad=0.1,h_pad=0.02, w_pad=0.2)
    # plt.show()
