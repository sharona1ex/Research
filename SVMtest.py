# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 11:23:46 2018

@author: SHALOM ALEXANDER
"""

from sklearn import  svm

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use("ggplot")
mapping_table = {
    (0,0,0,0) : -3-3j,
    (0,0,0,1) : -3-1j,
    (0,0,1,0) : -3+3j,
    (0,0,1,1) : -3+1j,
    (0,1,0,0) : -1-3j,
    (0,1,0,1) : -1-1j,
    (0,1,1,0) : -1+3j,
    (0,1,1,1) : -1+1j,
    (1,0,0,0) :  3-3j,
    (1,0,0,1) :  3-1j,
    (1,0,1,0) :  3+3j,
    (1,0,1,1) :  3+1j,
    (1,1,0,0) :  1-3j,
    (1,1,0,1) :  1-1j,
    (1,1,1,0) :  1+3j,
    (1,1,1,1) :  1+1j
}
map_table = {
(0,0):0,
(0,1):1,
(1,0):2,
(1,1):3        
}
demap_table = {
        0 : (0,0),
        1 : (0,1),
        2 : (1,0),
        3 : (1,1)
        }
def Mapping(bits):
    return np.array([map_table[tuple(b)] for b in bits])
#def Demap(bits):
#    return np.array([demap_table[tuple(b)] for b in bits])

x = np.array([[1,2],
             [9,9],
             [3,2],
             [1,0],
             [9,8],
             [10,11],
             [10,0],
             [9,2],
             [11,1],
             [2,10],
             [1,9],
             [0,10]])

plt.scatter(x[:,0],x[:,1])
plt.show()

y = [[0,0],[1,1],[0,0],[0,0],[1,1],[1,1],[0,1],[0,1],[0,1],[1,0],[1,0],[1,0]]

Y = Mapping(y)
print(Y)
Y2 = np.packbits(y,axis=-1)
#
clf = svm.SVC(kernel='linear',C=1.0)
clf.fit(x,Y)
#P = clf.predict([[12.0,9.0],[0,0],[9,9],[1,10]])
##print(P)
#p = Demap(P)
##print(p)