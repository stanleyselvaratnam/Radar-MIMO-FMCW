# test of matplotlib animation inside tk window

import sys
import numpy as np
import random
import matplotlibtk

def main(argv):

    def help(**kwargs):
        print(f"""
Usage: python3 {argv[0]}

Options:
  --help     display this message and exit
  --no-anim  no animation
  --thread   update plot from a separate thread
""", **kwargs)


    anim=True
    thr=False

    import getopt

    try:
        options, args = getopt.getopt(argv[1:], "", [
            "help",
            "no-anim",
            "thread",
        ])
        for name, value in options:
            if name == "--help":
                help()
                sys.exit(0)
            elif name == "--no-anim":
                anim = False
            elif name == "--thread":
                thr = True
    except Exception:
        help(file=sys.stderr)
        sys.exit(1)

    def plot(figure, data, ylim=None):
        figure.clear()
        ax = figure.add_subplot(111)
        ax.set_ylim(ylim or [-1, 1])
        ax.plot(data)

    done = False
    if anim:
        if thr:
            import threading, queue, time
            def thread(q):
                while not done:
                    q.put([0.2 * random.normalvariate() for _ in range(1024)])
                    time.sleep(0.1)
            def plot_from_queue(figure, q):
                try:
                    data = q.get_nowait()
                    data_np = np.array(data)
                    f = np.abs(np.fft.fft(data_np))
                    plot(figure, f, [0, 20])
                except queue.Empty:
                    pass
            q = queue.Queue()
            p = matplotlibtk.PlotWin(title="Animated plot with data from thread",
                        periodic_update_fun=plot_from_queue,
                        periodic_update_args=(q, ))
            t = threading.Thread(target=thread, args=(q,))
            t.start()
        else:
            p = matplotlibtk.PlotWin(title="Animated plot without thread",
                        periodic_update_fun=lambda f, ylim: plot(f, [random.normalvariate() for _ in range(100)], ylim),
                        periodic_update_args=([-5, 5], ),
                        period=0.01)
    else:
        p = matplotlibtk.PlotWin()
        p.call_update_fun(plot, ([random.normalvariate() for _ in range(100)], [-5, 5], ))
    
    p.mainloop()

    done = True


if __name__ == "__main__":
    main(sys.argv)
