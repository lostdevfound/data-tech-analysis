import numpy as np
from matplotlib import pyplot as pl
import sympy as sp

# compute coefficients for trig fourier series
def coef(vec, t, trigFunc):
    a = 0
    x = sp.Symbol('x')
    trigf = sp.cos(t*x)

    if trigFunc == "sin":
        trigf = sp.sin(t*x)
    elif trigFunc != "sin" and trigFunc != "cos":
        raise ValueError("trigFunc argument must be 'sin' or 'cos'")

    # iterate through the vector and get vec[i] and vec[i+1]
    for i in range(len(vec) - 1):
        scaler = 2*sp.pi / (len(vec) -1 )   # compute the scale value for compressing the domain into [0, 2pi]
        slope = (vec[i + 1] - vec[i]) / ((i + 1)*scaler - i*scaler)
        # put the data indexes into the domain of [0, 2pi]
        lowerBound = i*scaler
        upperBound = (i + 1)*scaler

        # compute the sum series of integrals with bounds i to i + 1
        if trigFunc == "cos":
            # TODO make it lambda function
            f = (-1)*i*scaler*slope*sp.sin(t*x)/t + vec[i]*sp.sin(t*x)/t + slope*x*sp.sin(t*x)/t + slope*sp.cos(t*x)/t**2

            if (t == 0):
                # TODO make it lmbda function
                f = slope*x**2/2 + x*(-i*scaler*slope + vec[i])
                a += (f.subs(x, upperBound) - f.subs(x, lowerBound)) * 1 / (2 * sp.pi)
            else:
                a += (f.subs(x, upperBound) - f.subs(x, lowerBound)) * 1 / sp.pi

        else:
            # TODO make it lambda function
            f = i*scaler*slope*sp.cos(t*x)/t - vec[i]*sp.cos(t*x)/t - slope*x*sp.cos(t*x)/t + slope*sp.sin(t*x)/t**2

            if (t == 0):
                a += 0
            else:
                a += (f.subs(x, upperBound) - f.subs(x, lowerBound)) * 1 / sp.pi
    # print(trigFunc, "coef:", t, a)
    return a


# compute the Fourier trigonometric series
def fourierTrigSeries(vec, n=10):
    x = sp.Symbol('x')
    series = 0
    scaler = 2*sp.pi / (len(vec) -1 )

    for t in range(n):
        series += coef(vec, t, "cos")*sp.cos(t*x*scaler) + coef(vec, t, "sin")*sp.sin(t*x*scaler)

    return sp.lambdify(x, series, modules=['numpy'])



xs = np.arange(0, 15, 0.1)
v = np.random.rand(15)
# v = [3, 3.5, 2, 2]
series = fourierTrigSeries(v, 6)
# pl.plot(xs, series(xs), 'r--')
# pl.plot(v, '-gD')
# pl.show()
