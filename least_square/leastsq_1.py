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
x = [0, 1, 2, 3, 4, 5, 6, 8]
y = [1, 2, 2, 2, 3, 3, 4, 4]
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

pl.scatter(x, y)
pl.plot(x, func(x, plsq[0]), label="fiited curve", marker="*")
pl.legend()
pl.show()
