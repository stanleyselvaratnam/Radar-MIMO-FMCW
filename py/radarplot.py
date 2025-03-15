# plot of 2D radar data as a pixmap in a sector
# yves.piguet@csem.ch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

def radarplot(angle, radius, data, ax, cmap=None,
              angle_ticks=None,
              radius_ticks=None,
              radius_unit=""):
    """
    Plot a 2D array of values between 0 and 1 as a radar plot, where data
    row i corresponds to angle[i] and column j to radius[j] (actually to
    patches between angle[i] and angle[i+1], or between radius[j] and
    radius[j+1], respectively).

    Parameters:
      angle: vector of angles, one more than rows in data
      radius: vector of radii, one more than columns in data
      data: 2D array with values in [0,1]
      ax: matplotlib axe
      cmap: matplotlib colormap, or None for blue shades
    Optional parameters:
      angle_ticks: list of angles where labels are displayed
      radius_ticks: list of radii where labels are displayed
      radius_unit: string appended to radius labels
    """
    
    na = len(angle)
    nr = len(radius)
    angle_a = np.radians(angle).reshape((-1, 1)).repeat(nr, axis=1)
    radius_a = radius.reshape((1, -1)).repeat(na, axis=0)
    x = (radius_a * np.sin(angle_a)).reshape((-1))
    y = (radius_a * np.cos(angle_a)).reshape((-1))
    # triangles: topleft, bottomleft, bottomright; then topleft, bottomright, topright
    tri = [
        [nr * i + j, nr * i + nr + j, nr * i + nr + j + 1]
        for i in range(na - 1)
        for j in range(nr - 1)
    ] + [
        [nr * i + j, nr * i + nr + j + 1, nr * i + j + 1]
        for i in range(na - 1)
        for j in range(nr - 1)
    ]
    triangulation = matplotlib.tri.Triangulation(x, y, triangles=tri)

    # flatten and repeat data for the two sets of triangles
    d = np.tile(data.reshape((-1)), 2)

    ax.tripcolor(triangulation, d,
                 cmap=cmap if cmap is not None else matplotlib.colormaps["Blues"])

    # text metric
    M_bb = (ax.text(0, 0, "MMMMMMMMMM", visible=False)
            .get_window_extent(renderer=ax.get_figure().canvas.get_renderer())
            .transformed(ax.transData.inverted()))

    if angle_ticks is not None:
        print("M_bb", M_bb)
        print("M_bb.width", M_bb.width)
        print("M_bb.height", M_bb.height)
        r = np.max(radius) + 15 * M_bb.height
        for a in angle_ticks:
            ax.text(r * math.sin(np.radians(a)), r * math.cos(np.radians(a)),
                    f"{a}°",
                    ha="center", va="bottom")

    if radius_ticks is not None:
        a = np.max(angle)
        for r in radius_ticks:
            ax.text(r * math.sin(np.radians(a)), r * math.cos(np.radians(a)),
                    f" {r}{radius_unit}",
                    ha="left", va="top")

    ax.axis("equal")
    ax.axis("off")
