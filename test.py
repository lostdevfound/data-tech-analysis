import numpy as np
import sympy as sp
from math import *
from matplotlib import pyplot as plt
import seaborn as sns
from tech_analysis import*

# test program
if __name__ == "__main__":
    sns.set_style("whitegrid")

    xs = np.arange(0, 25, 0.1)
    # v = np.random.rand(35)
    v = [ 0.51489764,  0.02507128,  4.30564675,  0.30564675,   -3.02507128,  0.59657168,
          0.31133694,  5.03578274,  0.1550626,   -3.04124578,  0.59126738,  0.2298406,
          0.9553957,   -1.93778872,  4.37510539,  -1.3242265,   0.68270655,  0.28223133,
          0.57876526,  -0.49377124,  1.84679945,  -2.38998178,  0.85667925,  0.12089524,
          0.54092031]
    plt.figure(1)
    plt.subplot(311)

    N = 20 # fourier series number of summations
    series, trigTerms = fourierTrigSeries(v, N)
    plt.title('Fourier Analysis')
    plt.plot(v, 'gs--', label='Original Data')
    plt.plot(xs, series(xs), '-b', label='Fourier Trig Series, n={}'.format(N))
    plt.legend(loc='lower left')

    plt.subplot(323)
    coef,trigfuncs = zip(*trigTerms)
    markerline, stemlines, baseline = plt.stem(coef)
    plt.xticks([x for x in range(len(coef))], trigfuncs, rotation=15)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.title('frequency amp')

    plt.subplot(324)
    x = sp.Symbol('x')
    plt.title('frequency decomposition')
    for i in range(len(trigfuncs)):
        if i < 2:
            continue
        f = sp.lambdify(x, trigfuncs[i], modules=['numpy'])
        line, = plt.plot(xs, coef[i]*f(xs))
        line.set_linewidth(0.8)


    smoothingFactor = 10
    convolvedSeries, convolvedTrigTerms = convolveFourierSeries(trigTerms, smoothingFactor)
    # emaVector = emaData(v, 0.5)
    plt.subplot(313)
    plt.title('Convolution with Gauss function in frequency space')
    plt.plot(v, 'gs--', label='Original Data')
    plt.plot(xs, convolvedSeries(xs), 'k', label='convolved series')
    plt.legend(loc='lower left')
    plt.tight_layout(pad=0.1,h_pad=0.02, w_pad=0.2)
    plt.show()
