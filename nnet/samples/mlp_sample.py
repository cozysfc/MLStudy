# coding:utf-8


import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlp import *
import numpy


x = numpy.array([[1,1,1,0,0,0],
                 [1,0,1,0,0,0],
                 [1,1,1,0,0,0],
                 [0,0,1,1,1,0],
                 [0,0,1,1,0,0],
                 [0,0,1,1,1,0]])

y = numpy.array([[1],
                 [1],
                 [1],
                 [0],
                 [0],
                 [0]])

layers = Layers()
layers.add(6, 4, tanh)
layers.add(4, 2, tanh)
layers.add(2, 1, sigmoid)

layers.build()

layers.fit(x, y, epoch=3000)


print "-------------------------------------"
clf = layers.pre()
print "output\t\tanswer"
for _x, _y in zip(x, y):
    print clf(_x), "\t", _y


print "-------------------------------------"
print "layers config"
layers()

