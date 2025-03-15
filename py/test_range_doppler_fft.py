# test of function range_doppler_fft in radaralgo.py
# yves.piguet@csem.ch

import radaralgo
import peaks2d
import sys
import numpy as np


B = 9e9  # f_max - f_min = 9 GHz
TC = B / 90e12  # ramp rate = 90 MHz / us
FREQ0 = 57e9  # freq = 57..66 GHz
TS = 1e-6  # sampling period = 1 us
DEFAULT_NUM_SEQUENCES = 256


def test_range_doppler_fft(argv):

    def help(**kwargs):
        print(f"""
Usage: python3 {argv[0]}

Options:
  --help                display this message and exit
  --n n                 number of sequences (default: {DEFAULT_NUM_SEQUENCES})
  --noise n             noise standard deviation on IQ samples (default: 0)
  --peaks               display all local peaks
  --plot                display range-doppler map
  --range r             range (default: 1)
  --range r1,r2,...     range for multiple targets
  --velocity v          velocity (default: 1)
  --velocity v1,v2,...  velocity for multiple targets
  --velocity-max vmax   maximum velocity magnitude displayed by --plot
                        (default: 1.5 * max(abs(v)))

Examples:
  python3 {argv[0]}
  python3 {argv[0]} --range 1.5,0.5,0.6,0.6 --velocity 3,-1,2,0 --peaks
  python3 {argv[0]} --n 8192
  python3 {argv[0]} --plot
  python3 {argv[0]} --n 256 --plot --velocity-max 5 --range 1.5,0.5,0.6,0.6 --velocity 3,-1,2,0
""", **kwargs)

    R = [1]
    velocity = [1]
    noise = 0
    n = DEFAULT_NUM_SEQUENCES
    do_peaks = False
    do_plot = False
    velocity_max = None

    import getopt

    try:
        options, args = getopt.getopt(argv[1:], "", [
            "help",
            "n=",
            "noise=",
            "peaks",
            "plot",
            "range=",
            "velocity=",
            "velocity-max=",
        ])
        for name, value in options:
            if name == "--help":
                help()
                sys.exit(0)
            elif name == "--n":
                n = int(value)
            elif name == "--noise":
                noise = float(value)
            elif name == "--peaks":
                do_peaks = True
            elif name == "--plot":
                do_plot = True
            elif name == "--range":
                R = [float(s) for s in value.split(",")]
            elif name == "--velocity":
                velocity = [float(s) for s in value.split(",")]
            elif name == "--velocity-max":
                velocity_max = float(value)
    except Exception:
        help(file=sys.stderr)
        sys.exit(1)
    
    # repeat last elements of R or velocity to match size
    if len(R) > len(velocity):
        velocity += (len(R) - len(velocity)) * velocity[-1]
    elif len(R) < len(velocity):
        R += (len(velocity) - len(R)) * R[-1]
    # default --velocity-max
    if velocity_max is None:
        velocity_max = 1.5 * max([abs(v) for v in velocity])

    gen = radaralgo.RadarAlgo(Tc=TC, B=B, freq0=FREQ0, Ts=TS)
    calc = radaralgo.RadarAlgo(Tc=TC, B=B, freq0=FREQ0, Ts=TS)
    
    # generate n sequences of samples and stack them as matrix rows
    # (sum over all R and velocity elements)
    samples = None
    for i in range(len(R)):
        # reset phase for each target (optional, arbitrary starting phase wouldn't harm anyway)
        gen.reset()

        samples_i = np.concatenate([
            gen.synthetic_samples(R=R[i],
                                  velocity=velocity[i],
                                  noise=noise)
            .reshape((1, -1))
            for _ in range(n)
        ], axis=0)
        if samples is None:
            samples = samples_i
        else:
            samples += samples_i

    # calculate range-doppler FFT
    f, scale_fun = calc.range_doppler_fft(samples)

    # find (highest) peak
    if do_peaks:
        # move v=0 to the middle of the array for smooth lp filtering and peak detection
        # keep R=0 at j=0 (neither target expected there nor negative R)
        f_shifted = np.fft.fftshift(f, axes=(0,))
        ff_shifted = peaks2d.lpfilter2(np.abs(f_shifted), radius=min(f.shape)/50).tolist()
        loc_shifted = peaks2d.peak2(ff_shifted)
        # undo effect of fftshift on loc
        loc = [
            (i - n // 2 if i >= n // 2 else i + n // 2, j)
            for i, j in loc_shifted
        ]
        print(f"{len(loc)} peaks in range-doppler FFT:")
        for peak_i, peak_j in loc:
            peak_range, peak_velocity = scale_fun(peak_i, peak_j)
            print(f"  R={peak_range}  v={peak_velocity}")
    else:
        peak_flat_index = np.argmax(np.abs(f))
        peak_i, peak_j = np.unravel_index(peak_flat_index, shape=f.shape)
        peak_range, peak_velocity = scale_fun(peak_i, peak_j)
        print("Highest peak in range-doppler FFT:")
        print(f"  i={peak_i}  j={peak_j}")
        print(f"  R={peak_range}  v={peak_velocity}")

    # plot map
    if do_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlim(0, scale_fun(0, f.shape[1] - 1)[0])
        ax.set_ylim(-velocity_max, velocity_max)
        if do_peaks:
            range_velocity = [
                scale_fun(peak_i, peak_j)
                for peak_i, peak_j in loc
            ]
            ax.plot([rv[0] for rv in range_velocity], [rv[1] for rv in range_velocity], 'o')
        else:
            range_a = []
            velocity_a = []
            mag_a = []
            for i in range(f.shape[0]):
                for j in range(f.shape[1]):
                    range_ij, velocity_ij = scale_fun(i, j)
                    range_a.append(range_ij)
                    velocity_a.append(velocity_ij)
                    mag_a.append(np.abs(f[i, j]))
            # print(f"Span of R: {min(range_a)} - {max(range_a)}")
            # print(f"Span of velocity: {min(velocity_a)} - {max(velocity_a)}")
            # print(f"Span of mag F: {min(mag_a)} - {max(mag_a)}")
            ax.tripcolor(range_a, velocity_a, mag_a/max(mag_a))
        ax.set_title("Range-Doppler map")
        ax.set_xlabel("Range")
        ax.set_ylabel("Velocity")
        plt.show()


if __name__ == "__main__":
    test_range_doppler_fft(sys.argv)
