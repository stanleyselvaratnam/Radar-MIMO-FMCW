# Smoothing of data
# yves.piguet@csem.ch

"""
Usage:

avg = Smoothing(n=10)
for data in stream:
    a = avg.add(data)
    if a is not None:
        # a is the mean of last n data
        ...

data can be a list of numbers or a numpy array. Items are cropped
to the minimum size if required, but their number of dimensions must
be the same.
Call avg.reset() to discard accumulated data.
Call avg.set_n(n) at any time.
"""

import math
import numpy as np
import scipy


class Smoothing:

    def apply(self, x):
        return x


class SmoothingRaisedCosine(Smoothing):

    def __init__(self, n):
        super().__init__()
        self.kernel = self.raised_cosine(n)
    
    def apply(self, x):
        return np.convolve(x, self.kernel, mode="same")

    @staticmethod
    def raised_cosine(n):
        a = [(i - (n - 1) / 2) * 2 * math.pi / n for i in range(n)]
        kernel = 1 + np.cos(a)
        return kernel / np.sum(kernel)


class SmoothingZeroDelayIIR(Smoothing):

    """
    Apply successively in both direction a first-order discrete-time filter
    xf(k) = p xf(k-1) + (1-p) x(k), where p = 1 - 1 / n and xf(-1) = x(0)
    (padtype="constant") or odd ("odd") for fft
    """

    def __init__(self, n, padtype="odd"):
        super().__init__()
        self.p = 1 - 1 / n
        self.padtype = padtype

    def apply(self, x):
        return scipy.signal.filtfilt(1 - self.p, [1, -self.p], x, padtype=self.padtype)
