import numpy as np
import pylab as pl

import os

os.chdir("results")

weak_results = np.genfromtxt('weak_labeled.csv', delimiter=',')
full_results = np.genfromtxt('full_labeled.csv', delimiter=',')

weak_results[0, 2] = 0.5
full_results[0, 1] = 0.5

y = np.mean(weak_results[:17, :6], axis=0)
y_min = np.min(weak_results[:17, :6], axis=0)
y_max = np.max(weak_results[:17, :6], axis=0)
x = np.array([0, 2, 4, 10, 25, 100])

full_y = np.mean(full_results[:17, :5], axis=0)
full_y_min = np.min(full_results[:17, :5], axis=0)
full_y_max = np.max(full_results[:17, :5], axis=0)
full_x = np.array([2, 4, 10, 25, 100])

pl.errorbar([0, 1, 2, 3, 4, 5], 1 - y, yerr=[y_max - y, y - y_min],
            label='+weak')
pl.errorbar([1, 2, 3, 4, 5], 1 - full_y,
            yerr=[full_y_max - full_y, full_y - full_y_min], label='full')
pl.xticks(np.arange(0, 6), x)
pl.title('C=0.01')
pl.xlabel('number of fully-labeled examples')
pl.ylabel('hamming loss')
pl.ylim([0, 1])
pl.xlim([-0.1, 5.1])
pl.legend(loc='lower right')
pl.show()
