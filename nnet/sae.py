# coding:utf-8


import numpy as np
import theano, theano.tensor as T
from collections import OrderedDict
from mlp import Layers
from tqdm import tqdm


def sigmoid(x):
    return T.nnet.sigmoid(x)


def tanh(x):
    return T.tanh(x)


def identify(x):
    return x


def encode(depth, variable, weight):

    encoder = variable

    for d in range(depth+1):
        if d == depth:
            encoder = T.dot(encoder, weight[d])
            print "T.dot(encoder, weight[d])"
        else:
            encoder = T.dot(encoder, weight[d])
            print "T.dot(encoder, weight[d])"

    return encoder


def decode(depth, encoder, weight):

    decoder = encoder

    for d in sorted(range(depth+1), reverse=True):
        if d == depth:
            decoder = T.dot(decoder, weight[d].T)
            print "T.dot(decoder, weight[d].T)"
        else:
            decoder = T.dot(decoder, weight[d].T)
            print "T.dot(decoder, weight[d].T)"

    return decoder


def cost(variable, decoder):
    # Negative Log Likelihood Function
    #-T.sum(variable*T.log(decoder)+(1-variable)*T.log(1-variable))
    return T.sum((decoder-variable)**2)/2


class AE:
    def __init__(self, layers, eta=0.001):
        self.depth = layers.depth
        self.variable = layers.variable
        self.weights = layers.weights
        self.encoders = OrderedDict()
        self.decoders = OrderedDict()
        self.cost = OrderedDict()
        self.updates = OrderedDict()
        self.f = OrderedDict()

        for d in range(self.depth):
            print "stack:", d
            self.encoders[d] = encode(d, self.variable, self.weights)
            self.decoders[d] = decode(d, self.encoders[d], self.weights)
            self.cost[d] = cost(self.variable, self.decoders[d])
            self.updates[d] = {
                self.weights[d]:self.weights[d]-eta*T.grad(self.cost[d], self.weights[d])
            }
            print ""

    def __call__(self):
        print "depth:", self.depth
        print "wights:", self.weights
        print "encoders:", self.encoders
        print "decoders:", self.decoders
        print "updates:", self.updates
        print "cost:", self.cost
        print "f:", self.f

    def build(self):
        for d in range(self.depth):
            self.f[d] = theano.function([self.variable], self.cost[d], updates=self.updates[d])

    def fit(self, X, epoch=100):
        epoch = xrange(epoch)
        for f in tqdm(self.f):
            for e in epoch:
                for x in X:
                    self.f[f](x)

    def train(self, X, epoch=5):
        epoch = xrange(epoch)
        for f in self.f:
            for e in epoch:
                for x in tqdm(X):
                    self.f[f](x)
