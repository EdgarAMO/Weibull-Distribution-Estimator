
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:00:00 2022
@author: Edgar
"""
#......Open the sample file:
import numpy as np
v, f = np.loadtxt('sample.csv', delimiter=',', unpack=True)

#......Turn percentages into frequencies:
f = f/100.

#......Visualize data sample:
print('**v*     ***f**')
for i in zip(v, f):
    print('{0:4.1f}\t {1:5.4}'.format(*i))
    
#......Gamma function:
from scipy.special import gamma
    
#......MMLM:
def mmlm(_v, _f):
    g = 1.
    while True:
        h1 = np.log(_v)*_v**g
        t1 = np.sum(h1*_f)
        t2 = np.sum(_f*(_v**g))
        t3 = np.sum(_f*np.log(_v))
        t4 = 1.
        n = (t1/t2 - t3/t4)**(-1)
        if abs((g - n)/n) < 0.01:
            break
        else:
            g = n
    _c = (t2/t4)**(1/n)
    return n, _c

#......Weibull parameters
k, c  = mmlm(v, f)
print('shape parameter: {0:5.3f}, scale parameter: {1:5.3f}'.format(k, c))
    
    
#......Sample data:
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(1,1,1)
ax.plot(v, f, marker='D', markersize=8, markerfacecolor='w')
ax.grid()
ax.set_xlim(0,14)
ax.set_xlabel('velocity bins')
ax.set_ylabel('frequency')
plt.show()

#......Weibull function:
def weibull(_x, _k, _c):
    p = ((_k/_c)*(_x/_c)**(_k-1))*np.exp(-(_x/_c)**_k)
    return p

#......Embedded charts:
weib = plt.figure(figsize=(6,6))
axis = weib.add_subplot(1,1,1)
x = np.arange(14)
axis.plot(x, weibull(x, k, c), color='blue', label='mmlm', marker='o')
axis.legend()
axis.grid()
axis.set_xlabel(r'wind speed $[m/s]$')
axis.set_ylabel(r'frequency [-]')
axis.set_facecolor('xkcd:silver')
axis.plot(v, f, 'b^', color='black', label='sample data', marker='s')
axis.legend()
plt.show()
        
