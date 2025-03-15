# test of radaralgo.py
# yves.piguet@csem.ch

import radaralgo
import sys
import getopt
import numpy as np

B = 9e9  # f_max - f_min = 9 GHz
TC = B / 90e12  # ramp rate = 90 MHz / us
FREQ_MIN = 57e9  # freq = 57..66 GHz
TS = 1e-6


def test_radaralgo(argv):

    def help(**kwargs):
        print(f"""
Usage: python3 {argv[0]}

Options:
  --help        display this message and exit
  --angle a     angle in degree (default: no angle)
  --direct      use direct algorithm (without IFFT) to generate synthetic samples
  --noise n     noise standard deviation on IQ samples (default: 0)
  --n n         perform n iterations (default: 10)
  --quiet       no output
  --range r     range (default: 1)
  --velocity v  velocity (default: 0)
""", **kwargs)

    R = 1
    velocity = 0
    angle = None
    noise = 0
    n = 10
    algo_gen_direct = False
    quiet = False

    import getopt

    try:
        options, args = getopt.getopt(argv[1:], "", [
            "help",
            "angle=",
            "direct",
            "n=",
            "noise=",
            "quiet",
            "range=",
            "velocity=",
        ])
        for name, value in options:
            if name == "--help":
                help()
                sys.exit(0)
            elif name == "--angle":
                angle = np.radians(float(value))
            elif name == "--n":
                n = int(value)
            elif name == "--noise":
                noise = float(value)
            elif name == "--direct":
                algo_gen_direct = True
            elif name == "--once":
                once = True
            elif name == "--quiet":
                quiet = True
            elif name == "--range":
                R = float(value)
            elif name == "--velocity":
                velocity = float(value)
    except Exception:
        help(file=sys.stderr)
        sys.exit(1)

    gen = radaralgo.RadarAlgo(Tc=TC, B=B, freq_min=FREQ_MIN, Ts=TS)
    calc = radaralgo.RadarAlgo(Tc=TC, B=B, freq_min=FREQ_MIN, Ts=TS)
    
    if angle is None:
        for _ in range(n):
            if algo_gen_direct:
                vec = gen.synthetic_samples_direct(R=R,
                                                   velocity=velocity,
                                                   noise=noise)
            else:
                vec = gen.synthetic_samples(R=R,
                                            velocity=velocity,
                                            noise=noise)
            R, v = calc.estimate_range_velocity(vec, True)
            if not quiet:
                print("R", R, "v", v)
    else:
        for _ in range(n):
            vec1 = gen.synthetic_samples(R=R,
                                         velocity=velocity,
                                         noise=noise)
            vec2 = gen.synthetic_samples(R=R,
                                         velocity=velocity,
                                         alpha=angle,
                                         noise=noise)
            R, alpha, v = calc.estimate_range_angle_velocity(vec1, vec2, True)
            if not quiet:
                print("R", R, "alpha", np.degrees(alpha), "v", v)

if __name__ == "__main__":
    test_radaralgo(sys.argv)
