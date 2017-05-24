# coding:utf-8


import numpy as np
import theano, theano.tensor as T

try:
    import cPickle as pickle
except:
    import pickle

from collections import OrderedDict
from tqdm import tqdm


def sigmoid(x):
    return T.nnet.sigmoid(x)


def tanh(x):
    return T.tanh(x)


def identify(x):
    return x


class Layers:
    def __init__(self, alpha=0.01):
        """
        This function will build multiple layer perceptron symbolically.

        Usage
        ----------
        build new layers:
            layers = Layers()
        add layers:
            layers.add(784, 500, tanh)
            layers.add(500, 200, tanh)
            # logistic regression
            layers.add(200, 10, sigmoid)
        train layers:
            layers.fit(<input>, <target>)
        use model:
            clf = layers.fprop()
            clf(<data>)
            => array([ 0.1131,  0.8224, ... ,  0.0833])
        see config:
            layers()
            => depth: 3, alpha: 0.01, activation: sigmoid,
               weights: OrderedDict([(0, w0), (1, w1), (2, w2)])
               updates: OrderedDict([(w0, Elemwise{sub,no_inplace}.0),
                                     (w1, Elemwise{sub,no_inplace}.0),
                                     (w2, Elemwise{sub,no_inplace}.0)])
        estimate loss(example):
            rss = layers.bf
        """
        self.depth = 0
        self.activation = OrderedDict()
        self.alpha = alpha
        self.variable = T.dvector('x')
        self.weights = OrderedDict()
        self.linear = OrderedDict()
        self.output = None
        self.updates = OrderedDict()

    def __call__(self):
        print "depth: %s, alpha: %s"%(self.depth, self.alpha)
        print "activation: %s"%self.activation
        print "weights: %s"%self.weights
        print "updates: %s"%self.updates

    def add(self, n_in, n_out, activation):
        self.activation[self.depth] = activation

        self.weights[self.depth] = theano.shared(
            name='w%s'%self.depth,
            value=np.random.uniform(-1.0, 1.0, (n_in, n_out)),
            borrow=True)

        if self.depth:
            self.linear[self.depth] = activation(T.dot(self.linear[self.depth-1], self.weights[self.depth]))
        else:
            self.linear[self.depth] = activation(T.dot(self.variable, self.weights[self.depth]))

        self.output = self.linear[self.depth]
        self.depth+=1

    def build(self):
        self.target = T.dvector('y')
        self.cost = T.sum((self.target-self.output)**2)/2
        self.updates = OrderedDict()
        for d in xrange(self.depth):
            self.updates[self.weights[d]] = (
                self.weights[d] - self.alpha * T.grad(self.cost, self.weights[d])
            )

        self.train = theano.function(
            inputs=[self.variable, self.target],
            outputs=self.cost,
            updates=self.updates
        )

    def fit(self, X, Y, alpha=0.01, epoch=100):
        X = np.array(X)
        Y = np.array(Y)
        print "train for %s epoch."%epoch
        for ep in tqdm(xrange(epoch)):
            for x, y in zip(X, Y):
                self.train(x, y)

        print "done."

    def pre(self):
        self.pf = theano.function([self.variable], self.output)
        return self.pf

    def save(self, model_name):
        with open(model_name, "wb") as wfp:
            pickle.dump(self, wfp)

    @staticmethod
    def load(model_name):
        with open(model_name, "rb") as rfp:
            ret = pickle.load(rfp)

        return ret
