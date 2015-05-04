import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import leastsq

x1 = np.array([0.0, 4.1, 8.2, 12.1, 16.1,
               20.1, 23.9, 28.0, 32.1, 35.9,
               40.0, 44.0, 48.0, 52.1, 56.1,
               60.1])

x2 = np.array([60.1, 64.0, 68.0, 71.9, 75.9,
               80.0, 84.0, 88.0, 92.0, 96.0,
               100.1, 104.1, 108.0, 112.1, 116.0,
               120.0])

x3 = np.array([120.0, 140.1, 160.1, 179.9, 200.0,
               249.9, 300.1, 350.0, 400.1, 450.0,
               497.7])

y1 = np.array([0, 3.73e-4, 1.68e-3, 4.10e-3, 6.89e-3,
              1.08e-2, 1.51e-2, 2.03e-2, 2.64e-2, 3.26e-2,
              3.97e-2, 4.71e-2, 5.51e-2, 6.37e-2, 7.25e-2,
              8.18e-2])

y2 = np.array([8.18e-2, 9.05e-2, 9.95e-2, 0.108, 0.116,
               0.123, 0.128, 0.133, 0.137, 0.141,
               0.145, 0.148, 0.152, 0.155, 0.159,
               0.162])

y3 = np.array([0.178, 0.195, 0.195, 0.211, 0.227,
               0.267, 0.308, 0.351, 0.397, 0.444,
               0.490])


def residuals(func):
    def ret(p, y, x):
        return y - func(x, p)
    return ret


def line(x, p):
    k, b = p
    return k * x + b


def square(x, p):
    a, b, c = p
    return a * x * x + b * x + c

# first stage
p = (1, 1, 1)
plsq = leastsq(residuals(square), p, args=(y1, x1))
pl.scatter(x1, y1)
pl.plot(x1, square(x1, plsq[0]), marker="*", label='%s*x^2+%s*x%s' % tuple(plsq[0]))
print(plsq)

# second stage
p = (1, 1, 1)
plsq = leastsq(residuals(square), p, args=(y2, x2))
pl.scatter(x2, y2)
pl.plot(x2, square(x2, plsq[0]), marker="*", label='%s*x^2+%s*x%s' % tuple(plsq[0]))
print(plsq)

# third stage
p = (1, 1)
plsq = leastsq(residuals(line), p, args=(y3, x3))
pl.scatter(x3, y3)
pl.plot(x3, line(x3, plsq[0]), marker="*", label='%s*x+%s' % tuple(plsq[0]))
print(plsq)

# to divide the area
x = np.array([60, 60])
y = np.array([-0, 0.5])
pl.plot(x, y, linestyle=":")

x = np.array([120, 120])
y = np.array([-0, 0.5])
pl.plot(x, y, linestyle=":")


# paint
pl.legend(loc="lower right")
pl.show()
