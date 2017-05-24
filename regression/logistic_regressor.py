# coding:utf-8


import os
import sys
import random

import theano
import theano.tensor as T
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

from collections import OrderedDict
from tqdm import tqdm
from pprint import pprint as pp

#from fastopt.utils import *


def sigmoid(x):
    return T.nnet.sigmoid(x)


def softmax(x):
    return T.nnet.softmax(x)


def tanh(x):
    return T.tanh(x)


def identify(x):
    return x


def shared(dataset, borrow=True):
    X_train, t_train = dataset

    X_train_total = X_train.shape[0]
    t_train_total = t_train.shape[0]

    if X_train_total == t_train_total:
        total = X_train_total
    else:
        raise SystemExit

    shared_X_train = theano.shared(np.asarray(X_train,
                                        dtype=np.float64),
                            borrow=borrow)
    shared_t_train = theano.shared(np.asarray(t_train,
                                        dtype=np.int64),
                            borrow=borrow)

    return total, shared_X_train, shared_t_train


def load_dataset(fpath, X_scale=(1.0, 0.0), m=None):
    """
    If train doesn't make sense, valiadate this function
    """
    with open(fpath) as rfp:
        dataset = pickle.load(rfp)

    X_train, t_train = dataset
    if m:
        sample = random.sample(range(len(X_train)), m)
        X_train = [X_train[i] for i in sample]
        t_train = [t_train[i] for i in sample]

    high, low = X_scale
    X_train = Scale(X_train, high=high, low=low).scaled

    return X_train, t_train


class LogisticRegressor:
    def __init__(self, input_n, output_n, X_train, t_train, batch=10, epoch=10, alpha=0.01, beta=0.01,
            activate=softmax, show_log=True):
        self.input_n = input_n
        self.output_n = output_n
        self.X_train = X_train
        self.t_train = t_train
        self.total = self.X_train.get_value().shape[0]
        self.batch = batch
        self.epoch = epoch
        self.alpha = alpha
        self.beta = beta
        self.batch = batch
        self.activate = activate
        self.variable = T.dmatrix("variable")
        self.target = T.lvector("target")
        self.idx = T.lscalar()

        self.trainsets = theano.function(
            inputs=[self.idx],
            outputs=[
                self.X_train[self.idx*self.batch:(self.idx+1)*self.batch],
                self.t_train[self.idx*self.batch:(self.idx+1)*self.batch]
            ]
        )

        self.show_log = show_log
        if self.show_log:
            print "LogisticRegressor.__init__: compile weights"

        self.weights = theano.shared(
            name="weights",
            value=np.random.random([input_n, output_n])
        )

        if self.show_log:
            print "LogisticRegressor.__init__: compile bias"

        self.bias = theano.shared(
            name="bias",
            value=np.random.random(output_n)
        )

    def build(self):
        self.o = self.activate(T.dot(self.variable, self.weights) + self.bias)
        self.p = T.argmax(self.o, axis=1)

        # log likelihood
        self.cost = -T.mean(
            T.log(self.o)[
                T.arange(self.target.shape[0]),
                self.target
            ]
        )

        self.updates = OrderedDict(
            {
                self.weights: self.weights-self.alpha*T.grad(self.cost, self.weights),
                self.bias: self.bias-self.beta*T.grad(self.cost, self.bias)
            }
        )

        if self.show_log:
            print "LogisticRegressor.build: compile train"

        self.train = theano.function(
            inputs=[self.idx],
            outputs=self.cost,
            updates=self.updates,
            givens={
                self.variable:self.X_train[self.idx*self.batch:(self.idx+1)*self.batch],
                self.target:self.t_train[self.idx*self.batch:(self.idx+1)*self.batch]
            }
        )

        if self.show_log:
            print "LogisticRegressor.build: compile get_cost"

        self.get_cost = theano.function(
            inputs=[self.variable, self.target],
            outputs=self.cost
        )

        self.predict = theano.function(
            inputs=[self.variable],
            outputs=self.p
        )

    def flush(self):
        if self.show_log:
            print "LogisticRegressor.flush: init weights, bias"

        self.weights.set_value(np.random.random([input_n, output_n]))
        self.bias.set_value(np.random.random(output_n))

    def fit(self, show=True):
        epoch = tqdm(range(self.epoch))
        for e in epoch:
            for b in range(self.total/self.batch):
                self.train(b)

    def pre(self):
        if self.show_log:
            print "LogisticRegressor.pre: compile predictor"

        return self.predict

    def save(self, fpath="logreg.pickle"):
        if self.show_log:
            print "LogisticRegressor.save: dump model"

        with open(fpath, "wb") as wfp:
            pickle.dump(self, wfp)

    @staticmethod
    def load(fpath="logreg.pickle"):
        with open(fpath, "rb") as rfp:
            ret = pickle.load(rfp)

        return ret
