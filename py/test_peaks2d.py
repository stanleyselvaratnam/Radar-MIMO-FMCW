# test of radaralgo.py
# yves.piguet@csem.ch

import peaks2d
import numpy as np
import matplotlib.pyplot as plt
import sys

DEFAULT_SIZE = 100

def make_array(size=DEFAULT_SIZE, noise=0):
    x, y = np.meshgrid(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01))
    a = np.exp(-(x-0.3)**2/0.01-(y-0.6)**2/0.01) + np.exp(-(x-0.8)**2/0.005-(y-0.4)**2/0.005)
    if noise > 0:
        a += noise * np.random.normal(size=a.shape)
    return x, y, a

def test_peaks(x, y, a, n=1, filter_radius=0):
    for _ in range(n):
        if filter_radius > 0:
            aa = peaks2d.lpfilter2(a, radius=filter_radius).tolist()
        else:
            aa = a.tolist()
        i = peaks2d.peak2(aa)
    print("i", i)

def main(argv):

    def help(**kwargs):
        print(f"""
Usage: python3 {argv[0]}

Options:
  --help        display this message and exit
  --filter n    low-pass filter radius (default: 0)
  --noise n     noise standard deviation (default: 0)
  --n n         perform n iterations (default: 10)
  --size s      size (default: {DEFAULT_SIZE})
""", **kwargs)

    size = DEFAULT_SIZE
    noise = 0
    filter_radius = 0
    n = 10
    do_plot = False

    import getopt

    try:
        options, args = getopt.getopt(argv[1:], "", [
            "help",
            "filter=",
            "n=",
            "noise=",
            "plot",
            "size=",
        ])
        for name, value in options:
            if name == "--help":
                help()
                sys.exit(0)
            elif name == "--filter":
                filter_radius = float(value)
            elif name == "--n":
                n = int(value)
            elif name == "--noise":
                noise = float(value)
            elif name == "--plot":
                do_plot = True
            elif name == "--size":
                size = int(value)
    except Exception:
        help(file=sys.stderr)
        sys.exit(1)
    
    x, y, a = make_array(size=size, noise=noise)

    if do_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, a)
        plt.show()
    
    test_peaks(x, y, a, n=n, filter_radius=filter_radius)

if __name__ == "__main__":
    main(sys.argv)
