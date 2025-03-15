# Local maxima in 2D array
# yves.piguet@csem.ch

import math

def peak1(a, include_ends=False):
    """
    Find local maxima of 1D list a.
    """

    n = len(a)
    if n == 0:
        return []
    elif n == 1:
        return [0] if include_ends else []
    if include_ends:
        return [
            i
            for i in range(n)
            if (i == 0 or a[i] > a[i - 1]) and (i == n - 1 or a[i] > a[i + 1])
        ]
    else:
        return [
            i
            for i in range(1, n - 1)
            if a[i] > a[i - 1] and a[i] > a[i + 1]
        ]

def valley1_val(a, p):
    """
    Find lowest values between peaks, including before first one
    and after last one
    """

    if len(p) == 0:
        return []

    # before first peak
    valley = [min(a[:p[0]]) if p[0] > 0 else a[p[0]]]
    # between peaks
    for i in range(len(p) - 1):
        valley.append(min(a[p[i] : p[i + 1]]))
    # after last peak
    valley.append(min(a[p[-1] + 1 : ]) if p[-1] < len(a) - 1 else a[p[-1]])

    return valley

def peak1_relative_height(a, p):
    """
    Find lowest values between peaks and return for each peak p
    the ratio p / max(lowest_left, lowest_right)
    """

    valley = valley1_val(a, p)

    # calculate relative heights
    relative_height = [
        a[p[i]] / max(valley[i], valley[i + 1])
        for i in range(len(p))
    ]
    return relative_height

def peak1_best(a, n=None, relative_height_threshold=None):
    """
    Find the n "best" peaks in 1D array:
        find all local maxima
        while more than n:
            find index of peak with smallest val_peak/max(val_valley_left, val_valley_right)
            remove index of peak
            merge val_valley_left, val_valley_right as min(val_valley_left, val_valley_right)
    Parameters:
      a  input values in a list
      n  number of peaks to keep (or above relative_height); default is None to keep all
      relative_height_threshold: min relative height wrt. left and right minima; default is 0
    """

    p = peak1(a)
    if n is None and relative_height_threshold is None:
        return p
    v = valley1_val(a, p)
    np = len(p)
    if n is None:
        n = 0
    if relative_height_threshold is None:
        relative_height_threshold = 0
    while np > n:
        relative_height = [
            a[p[i]] / max(max(v[i], v[i + 1]), 1e-10)
            for i in range(np)
        ]
        min_relative_height = min(relative_height)
        if min_relative_height >= relative_height_threshold:
            break
        ix_min_relative_height = relative_height.index(min_relative_height)
        del p[ix_min_relative_height]
        v_new = min(v[ix_min_relative_height], v[ix_min_relative_height + 1])
        del v[ix_min_relative_height + 1]
        v[ix_min_relative_height] = v_new
        np -= 1
    return p

def peak2(a, include_ends=False):
    """
    Find local maxima of 2D list of lists a.
    """

    a1_max = [max(a1) for a1 in a]
    p1 = peak1(a1_max, include_ends=include_ends)
    a_min = min(min(a1) for a1 in a)
    a_max = max(a1_max)
    a_threshold = 0.9 * a_min + 0.1 * a_max
    loc = [
        (i, j)
        for i in p1
        for j in peak1(a[i], include_ends=include_ends)
        if a[i][j] > a_threshold
    ]
    loc_s = set(
        opt2(a, i, j, stop_on_boundary = not include_ends)
        for i, j in loc
    )
    return list(loc_s)

def opt2(a, i, j, stop_on_boundary=True):
    """
    Ascend from (i, j) to nearest local maximum a[i][j]
    """
    offset = 0
    if not stop_on_boundary:
        # extend with 0
        import numpy as np
        a = np.pad(a, [1, 1], constant_values=-999999999)
        i += 1
        j += 1
        offset = 1
    while True:
        if i == 0 or i == len(a) - 1 or j == 0 or j == len(a[0]) - 1:
            return i, j
        a_ij = a[i][j]
        if a[i - 1][j] > a_ij:
            i -= 1
        elif a[i - 1][j - 1] > a_ij:
            i -= 1
            j -= 1
        elif a[i - 1][j + 1] > a_ij:
            i -= 1
            j += 1
        elif a[i][j - 1] > a_ij:
            j -= 1
        elif a[i][j + 1] > a_ij:
            j += 1
        elif a[i + 1][j] > a_ij:
            i += 1
        elif a[i + 1][j - 1] > a_ij:
            i += 1
            j -= 1
        elif a[i + 1][j + 1] > a_ij:
            i += 1
            j += 1
        else:
            break
    return i - offset, j - offset

def lpfilter2(a, radius):
    """
    Apply a low-pass filter to 2D list of lists a.
    """
    import scipy
    return scipy.ndimage.gaussian_filter(a, sigma=radius)
