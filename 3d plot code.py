# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 13:43:33 2020

@author: xf290
"""

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np


# The skiprows argument is because the first row is dataset number
literature_data = pd.read_csv('Feed In Data.csv', skiprows=1)

# Extract dataset 3 from the complete dataset frame
literature_data = literature_data.drop(columns=['1author2', '1author3', 'LiTFSIwt3', 'temp C3', 'exponent3', 'Unnamed: 4', 'LiTFSIwt2', 'temp C2', 'exponent2', 'Unnamed: 9', '1author'])

# Separate the dataset into training and test sets
literature_data = literature_data.dropna()

# Fixing random state for reproducibility
np.random.seed(19680801)


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zlow, zhigh)
    ax.scatter(xs, ys, zs, marker=m)

ax.set_xlabel('LiTFSI wt%')
ax.set_ylabel('Temperature (C)')
ax.set_zlabel('Conductivity log')

plt.show()