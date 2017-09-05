import numpy as np
import sympy as sp
from math import *
from matplotlib import pyplot as plt
import seaborn as sns
from tech_analysis import*

# test program
if __name__ == "__main__":
    sns.set_style("whitegrid")


    xs = np.arange(0, 15, 0.1)
    v = np.random.rand(15)
    plt.figure(1)
    plt.subplot(311)
    series, trigTerms = fourierTrigSeries(v, 8)
    plt.title('Fourier Analysis')
    plt.plot(v, 'gs--', label='Original Data')
    plt.plot(xs, series(xs), '-b', label='Fourier Trig Series, n=8')
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

    emaVector = emaData(v, 0.5)
    plt.subplot(313)
    plt.title('Exponential Moving Average')
    plt.plot(v, 'gs--', label='Original Data')
    plt.plot(emaVector, 'k', label='EMA')
    plt.legend(loc='lower left')
    plt.tight_layout(pad=0.1,h_pad=0.02, w_pad=0.2)
    plt.show()
