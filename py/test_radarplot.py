# test of radarplot.py
# yves.piguet@csem.ch

import matplotlib.pyplot as plt
import numpy as np

import radarplot

# make data: 2D np-array, row i = angle[i] (0=up), column j = radius
angle_min = -32
angle_max = 32
angle_step = 4
radius_min = 10
radius_max = 250
radius_step = 2
angle = np.array(range(angle_min, angle_max + angle_step // 2, angle_step))
radius = np.array(range(radius_min, radius_max + radius_step // 2, radius_step))
angle_num = len(angle)
radius_num = len(radius)
data = np.random.uniform(0, 1, (angle_num - 1, radius_num - 1))

fig, ax = plt.subplots()
radarplot.radarplot(angle, radius, data, ax,
                    angle_ticks=[-30, 0, 30],
                    radius_ticks=[10, 50, 100, 150, 200, 250],
                    radius_unit=" cm")
plt.show()
