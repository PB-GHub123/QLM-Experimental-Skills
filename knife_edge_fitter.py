import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from math import floor, log10


# Test test commit
def beam(x, x0, w, amp, offset):
    Phi = lambda x: 1 / 2 * (1 + erf(x))
    return amp * Phi(np.sqrt(2) * (x - x0) / w) + offset


def gaussian(x, x0, w):
    return np.exp(-2 * (x - x0) ** 2 / w**2)


# %%
x = "distance (mm)"
y = "voltage (V)"

for trace in ["test_knife_edge"]:
    df = pd.read_csv(trace + ".csv")
    plt.scatter(df[x], df[y])
    popt, pcov = curve_fit(beam, df[x], df[y], p0=(5, 1, 15, 0))
    perr = np.sqrt(np.diag(pcov))
    xfit = np.linspace(min(df[x]), max(df[x]), 1000)
    plt.plot(xfit, beam(xfit, *popt), c="C1")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(trace)
    plt.show()

    print("\n== {} ==".format(trace))
    for arg, val, err in zip(beam.__code__.co_varnames[1:], popt, perr):
        prec = floor(log10(err))
        err = round(err / 10**prec) * 10**prec
        val = round(val / 10**prec) * 10**prec
        if prec > 0:
            valerr = "{:.0f}({:.0f})".format(val, err)
        else:
            valerr = "{:.{prec}f}({:.0f})".format(val, err * 10**-prec, prec=-prec)
        print(arg, "=", valerr)
