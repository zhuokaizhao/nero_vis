# The script visualizes the random rotation effect and vis as histogram
import numpy as np
from scipy.stats import truncnorm
from matplotlib import pyplot as plt


all_percentages = [33, 66, 100]
bins = list(range(-180, 190, 10))
colors = ['green', 'cyan', 'pink']

for i, percentage in enumerate(all_percentages):
    # von mises
    mu = 0
    if percentage == 33:
        kappa = 100/(4*percentage) * np.pi
    elif percentage == 66:
        kappa = 100/(5*percentage) * np.pi
    elif percentage == 100:
        kappa = 100/(8*percentage) * np.pi

    all_rots = np.random.vonmises(mu, kappa, 10000) / np.pi * 180

    # # truncated gaussian
    # sigma = 360*percentage/200/2
    # # distribution that has the support as the rotation limit
    # a = (-180 - 0) / sigma
    # b = (180 - 0) / sigma

    # all_rots = []

    # for i in range(10000):
    #     # random rotation
    #     rot = float(truncnorm.rvs(a, b, scale=sigma, size = 1)[0])
    #     all_rots.append(rot)

    n, x, _ = plt.hist(all_rots, bins=bins, alpha=0.7, color=colors[i])
    bin_centers = 0.5 * (x[1:] + x[:-1])
    plt.plot(bin_centers, n, label=f'{percentage} rotation', color=colors[i])


plt.legend()
plt.show()
