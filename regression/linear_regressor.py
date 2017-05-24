# coding:utf-8


import theano
import theano.tensor as T
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

from collections import OrderedDict
from tqdm import tqdm


class LinearRegressor:
    def __init__(self, input_n, epoch=10, alpha=0.01, beta=0.01, show_log=True):
        """
        Modified for iris predict experiment.
        """
        self.input_n = input_n
        self.epoch = epoch
        self.alpha = alpha
        self.beta = beta
        self.variable = T.dvector("variable")
        self.target = T.dscalar("target")

        self.show_log = show_log
        if self.show_log:
            print "LinearRegressor.__init__: compile weights"

        self.weights = theano.shared(
            name="weights",
            value=np.random.random(input_n)
        )

        if self.show_log:
            print "LinearRegressor.__init__: compile bias"
        self.bias = theano.shared(
            name="bias",
            value=np.random.random()
        )

    def build(self):
        self.output = T.dot(self.weights, self.variable)+self.bias
        self.cost = T.mean((self.target-self.output)**2)

        self.updates = OrderedDict(
            {
                self.weights: self.weights-self.alpha*T.grad(self.cost, self.weights),
                self.bias: self.bias-self.beta*T.grad(self.cost, self.bias)
            }
        )


        if self.show_log:
            print "LinearRegressor.build: compile train"

        self.train = theano.function(
            inputs=[self.variable, self.target],
            outputs=self.cost,
            updates=self.updates
        )

        if self.show_log:
            print "LinearRegressor.build: compile get_cost"

        self.get_cost = theano.function(
            inputs=[self.variable, self.target],
            outputs=self.cost
        )

    def flush(self):
        if self.show_log:
            print "LinearRegressor.flush: init weights, bias"

        self.weights.set_value(np.random.random(self.input_n))
        self.bias.set_value(np.random.random())

    def fit(self, X, Y):
        epoch = tqdm(range(self.epoch))
        for e in epoch:
            for x, y in zip(X, Y):
                self.train(x, y)

    def pre(self):
        if self.show_log:
            print "LinearRegressor.pre: compile predictor"

        return theano.function([self.variable], self.output)

    def save(self, fpath="linreg.pickle"):
        if self.show_log:
            print "LinearRegressor.save: dump model"

        with open(fpath, "wb") as wfp:
            pickle.dump(self, wfp)

    @staticmethod
    def load(fpath="linreg.pickle"):
        if self.show_log:
            print "LinearRegressor.load: load model"

        with open(fpath, "rb") as rfp:
            ret = pickle.load(rfp)

        return ret
