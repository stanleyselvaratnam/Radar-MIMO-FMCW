# ratemeter.py
# yves.piguet@csem.ch

import math
import time

class RateMeter:

    def __init__(self, tau=1, n=1):
        self.tau = tau
        self.reset(n)
    
    def reset(self, n=1):
        self.rate = [0 for _ in range(n)]
        self.t = time.time()

    def notify(self, x):
        """
        x: vector of amounts since last call
        """
        t = time.time()
        delta = t - self.t
        self.t = t
        self.rate = [
            math.exp(-delta / self.tau) * rate1 + x1 / self.tau
            for x1, rate1 in zip(x, self.rate)
        ]
        # steady-state rate for constant delta and x and small delta: x / delta
    
    def get_rate(self):
        """
        Return vector of rates
        """
        t = time.time()
        return [
            rate1 * math.exp(-(t - self.t) / self.tau)
            for rate1 in self.rate
        ]

