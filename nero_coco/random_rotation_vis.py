# The script visualizes the random rotation effect and vis as histogram
import numpy as np
from scipy.stats import truncnorm
from matplotlib import pyplot as plt


import truncated_mvn_sampler


# multivariate gaussian (correctly truncated)
# for mnist rotate
# sigma = 90
# a = (-180 - 0) / sigma
# b = (180 - 0) / sigma
# r = np.array(truncnorm.rvs(a, b, scale=sigma, size = 10000))
# r = np.rint(r).astype(int)
# print(max(r), min(r))
# plt.hist(r, bins=range(-180, 181))
# plt.show()
# exit()

# for mnist shift
d = 2
mu = np.zeros(d)
cov = np.array([[100, 0], [0, 100]])
lb = np.zeros_like(mu) - 20
ub = np.zeros_like(mu) + 20
n_samples = 10000
tmvn = truncated_mvn_sampler.TruncatedMVN(mu, cov, lb, ub)
x, y = tmvn.sample(n_samples)
x = np.rint(x).astype(int)
y = np.rint(y).astype(int)
print(max(x), min(x))
print(max(y), min(y))
plt.hist2d(x, y, bins=range(-20, 21))
plt.show()
exit()

# for mnist scale
# sigma = 0.35
# all_s = []
# for i in range(10000):
#     s = 0
#     while s < 0.3 or s > 1.7:
#         s = np.random.normal(1.0, sigma, size=1)
#     all_s.append(s[0])
# plt.hist(all_s)
# plt.show()
# exit()


# comparison between different truncated multivariate Gaussian sampling
# multivariate gaussian (no truncation)
mean = [0, 0]
cov = [[100, 0], [0, 100]]
all_x = []
all_y = []
for i in range(10000):
    x, y = np.random.multivariate_normal(mean, cov, 1).T
    print(x, y)
    exit()
plt.hist2d(all_x, all_y, bins=range(-20, 21))
plt.show()
exit()


# combine two 1d truncated gaussian
# with gaussian, percentage is converted to different sigma value
sigma_x = 10
sigma_y = 10

# distribution that has the support as the image boundary
a = (-20 - 0) / sigma_x
b = (20 - 0) / sigma_y

left_padding = np.array(truncnorm.rvs(a, b, scale=sigma_x, size = 10000)).astype(int)
top_padding = np.array(truncnorm.rvs(a, b, scale=sigma_y, size = 10000)).astype(int)

plt.hist2d(left_padding, top_padding, bins=range(-20, 21), alpha=0.7)
plt.show()


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
