# The script visualizes the random rotation effect and vis as histogram
import numpy as np
from scipy.stats import truncnorm
from matplotlib import pyplot as plt


all_percentages = [33, 66, 100]
bins = list(range(0, 370, 10))

for percentage in all_percentages:
    sigma = 360*percentage/150/2
    # distribution that has the support as the rotation limit
    a = (0 - 0) / sigma
    b = (360 - 0) / sigma

    all_rots = []

    for i in range(10000):
        # random rotation
        rot = float(truncnorm.rvs(a, b, scale=sigma, size = 1)[0])
        all_rots.append(rot)

    plt.hist(all_rots, bins=bins, label=f'{percentage} rotation', alpha=0.3)
plt.legend()
plt.show()
