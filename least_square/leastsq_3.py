# -*- coding: utf-8 -*-
import re

import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as pl


def func(x, p):
    a2, a1, a0 = p
    return a2 * x * x + a1 * x + a0


def residuals(p, y, x):
    return y - func(x, p)

# hack
x = [15.7, 16.6, 16.7, 17.1, 17.4, 17.7, 17.9, 18.0, 18.2]
y = [16.9, 17.5, 17.5, 17.8, 18.2, 18.2, 18.5, 18.6, 18.7]
p = (0, 0, 0)
if (len(x) == 0):
    print("Total of samples:")
    n = int(input())

    for i in range(n):
        x0, y0 = tuple(map(int, re.split(r' *', input())))
        x.append(x0)
        y.append(y0)
x = np.array(x)
y = np.array(y)

plsq = leastsq(residuals, p, args=(y, x))
print(plsq)

pl.plot(x, y, "*", label="real")
pl.plot(x, func(x, plsq[0]), "or", label="fiited curve")
pl.legend(loc="center left")
pl.show()
