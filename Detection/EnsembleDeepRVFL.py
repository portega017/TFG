#-- coding: utf-8 --
#@Time : 2021/3/27 20:40
#@Author : HUANG XUYANG
#@Email : xhuang032@e.ntu.edu.sg
#@File : Ensemble_Deep_RVFL.py
#@Software: PyCharm


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
# Changes: Ignore (RuntimeWarning: overflow encountered in exp) and (RuntimeWarning: invalid value encountered in true_divide)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Changes finished

class EnsembleDeepRVFL(BaseEstimator, ClassifierMixin):
    """A ensemble deep RVFL classifier or regression.

    Attributes:
        n_nodes: An integer of enhancement node number.
        lam: A floating number of regularization parameter.
        w_random_vec_range: A list, [min, max], the range of generating random weights.
        b_random_vec_range: A list, [min, max], the range of generating random bias.
        random_weights: A Numpy array shape is [n_feature, n_nodes], weights of neuron.
        random_bias: A Numpy array shape is [n_nodes], bias of neuron.
        beta: A Numpy array shape is [n_feature + n_nodes, n_class], the projection matrix.
        activation: A string of activation name.
        n_layer: A integer, N=number of hidden layers.
        data_std: A list, store normalization parameters for each layer.
        data_mean: A list, store normalization parameters for each layer.
        same_feature: A bool, the true means all the features have same meaning and boundary for example: images.
        task_type: A string of ML task type, 'classification' or 'regression'.
    """
    # Changes: __init__ method must have default arguments
    # Any sklearn method should be capable of being instantiated without passing any arguments to it
    def __init__(self, n_nodes=80, lam=1, w_random_range=[-1,1], b_random_range=[0,1], n_layer=2, n_jobs=1, same_feature=False, activation='relu', task_type='classification'):
        
        # Changes: every keyword argument accepted by __init__ should correspond to an attribute on the instance
        # There should be no logic, not even input validation, and the parameters should not be changed
        
        #assert task_type in ['classification', 'regression'], 'task_type should be "classification" or "regression".'
        self.n_nodes = n_nodes
        self.lam = lam
        self.w_random_range = w_random_range
        self.b_random_range = b_random_range
        #self.random_weights = []
        #self.random_bias = []
        #self.beta = []
        #a = Activators()
        #self.activation = getattr(Activators(), act)
        self.activation = activation
        self.n_layer = n_layer
        self.n_jobs = n_jobs
        #self.data_std = [None] * self.n_layer
        #self.data_mean = [None] * self.n_layer
        self.same_feature = same_feature
        self.task_type = task_type

    def fit(self, X, y):
        """

        :param X: Training data.
        :param y: Training label.
        :param n_class: An integer of number of class. In regression, this parameter won't be used.
        :return: No return
        """
        
        # Changes: Logic can't be applied in the __init__ method, it should be applied in the fit method
        assert self.task_type in ['classification', 'regression'], 'task_type should be "classification" or "regression".'
        self.random_weights = []
        self.random_bias = []
        self.beta = []
        self.data_std = [None] * self.n_layer
        self.data_mean = [None] * self.n_layer
        a = Activators()
        self.activation = getattr(Activators(), self.activation) # Relu defaulted
        # Changes finish


        assert len(X.shape) > 1, 'Data shape should be [n, dim].'
        assert len(X) == len(y), 'Label number does not match data number.'
        assert len(y.shape) == 1, 'Label should be 1-D array.'
        
        n_sample = len(X)
        n_feature = len(X[0])
        n_class = len(np.unique(y))
        h = X.copy()
        X = self.standardize(X, 0)
        if self.task_type == 'classification':
            y = self.one_hot(y, n_class)
        else:
            y = y
        for i in range(self.n_layer):
            h = self.standardize(h, i)  # Normalization data
            self.random_weights.append(self.get_random_vectors(len(h[0]), self.n_nodes, self.w_random_range))
            self.random_bias.append(self.get_random_vectors(1, self.n_nodes, self.b_random_range))
            h = self.activation(np.dot(h, self.random_weights[i]) + np.dot(np.ones([n_sample, 1]),
                                                                                    self.random_bias[i]))
            d = np.concatenate([h, X], axis=1)

            h = d

            d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)

            if n_sample > (self.n_nodes + n_feature):
                self.beta.append(np.linalg.inv((self.lam * np.identity(d.shape[1]) + np.dot(d.T, d))).dot(d.T).dot(y))
            else:
                self.beta.append(d.T.dot(np.linalg.inv(self.lam * np.identity(n_sample) + np.dot(d, d.T))).dot(y))
                
        return self

    def predict(self, X, output_prob=False):
        """

        :param X: Predict data.
        :return: When classification, return vote result,  addition result and probability.
                 When regression, return the mean output of edrvfl.
        """
        n_sample = len(X)
        h = X.copy()
        X = self.standardize(X, 0)  # Normalization data
        outputs = []
        for i in range(self.n_layer):
            h = self.standardize(h, i)  # Normalization data
            h = self.activation(np.dot(h, self.random_weights[i]) + np.dot(np.ones([n_sample, 1]),
                                                                                    self.random_bias[i]))
            d = np.concatenate([h, X], axis=1)

            h = d

            d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
            outputs.append(np.dot(d, self.beta[i]))
        if self.task_type == 'classification':
            vote_res = [np.argmax(item, axis=1) for item in outputs]
            vote_res = list(map(np.bincount, list(np.array(vote_res).transpose())))
            vote_res = np.array(list(map(np.argmax, vote_res)))

            add_proba = self.softmax(np.sum(outputs, axis=0))
            add_res = np.argmax(add_proba, axis=1)
            # return vote_res, (add_res, add_proba)
            return vote_res
        
        elif self.task_type == 'regression':
            return np.mean(outputs, axis=0)

    def predict_proba(self, X, output_prob=False):
        """
    
        :param X: Predict data.
        :return: When classification, return vote result,  addition result and probability.
                 When regression, return the mean output of edrvfl.
        """
        n_sample = len(X)
        h = X.copy()
        X = self.standardize(X, 0)  # Normalization data
        outputs = []
        for i in range(self.n_layer):
            h = self.standardize(h, i)  # Normalization data
            h = self.activation(np.dot(h, self.random_weights[i]) + np.dot(np.ones([n_sample, 1]),
                                                                                    self.random_bias[i]))
            d = np.concatenate([h, X], axis=1)
    
            h = d
    
            d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
            outputs.append(np.dot(d, self.beta[i]))
        if self.task_type == 'classification':
            vote_res = [np.argmax(item, axis=1) for item in outputs]
            vote_res = list(map(np.bincount, list(np.array(vote_res).transpose())))
            vote_res = np.array(list(map(np.argmax, vote_res)))
    
            add_proba = self.softmax(np.sum(outputs, axis=0))
            # add_res = np.argmax(add_proba, axis=1)
            # return vote_res, (add_res, add_proba)
            return add_proba


    def eval(self, X, y):
        """

        :param X: Evaluation data.
        :param y: Evaluation label.
        :return: When classification return vote and addition accuracy.
                 When regression return MAE.
        """

        assert len(X.shape) > 1, 'Data shape should be [n, dim].'
        assert len(X) == len(y), 'Label number does not match data number.'
        assert len(y.shape) == 1, 'Label should be 1-D array.'

        n_sample = len(X)
        h = X.copy()
        X = self.standardize(X, 0)
        outputs = []
        for i in range(self.n_layer):
            h = self.standardize(h, i)  # Normalization data

            h = self.activation(np.dot(h, self.random_weights[i]) + np.dot(np.ones([n_sample, 1]),
                                                                                    self.random_bias[i]))
            d = np.concatenate([h, X], axis=1)

            h = d

            d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)

            outputs.append(np.dot(d, self.beta[i]))
        if self.task_type == 'classification':
            vote_res = [np.argmax(item, axis=1) for item in outputs]
            vote_res = list(map(np.bincount, list(np.array(vote_res).transpose())))
            vote_res = np.array(list(map(np.argmax, vote_res)))
            vote_acc = np.sum(np.equal(vote_res, y)) / len(y)

            add_proba = self.softmax(np.sum(outputs, axis=0))
            add_res = np.argmax(add_proba, axis=1)
            add_acc = np.sum(np.equal(add_res, y)) / len(y)

            return vote_acc, add_acc
        elif self.task_type == 'regression':
            pred = np.mean(outputs, axis=0)
            mae = np.mean(np.abs(pred - y))
            return mae

    @staticmethod
    def get_random_vectors(m, n, scale_range):
        x = (scale_range[1] - scale_range[0]) * np.random.random([m, n]) + scale_range[0]
        return x

    @staticmethod
    def one_hot(x, n_class):
        y = np.zeros([len(x), n_class])
        for i in range(len(x)):
            y[i, int(x[i])] = 1 # Changes: Cast x to int to avoid index error
        return y

    def standardize(self, x, index):
        if self.same_feature is True:
            if self.data_std[index] is None:
                self.data_std[index] = np.maximum(np.std(x), 1/np.sqrt(len(x)))
            if self.data_mean[index] is None:
                self.data_mean[index] = np.mean(x)
            return (x - self.data_mean[index]) / self.data_std[index]
        else:
            if self.data_std[index] is None:
                self.data_std[index] = np.maximum(np.std(x, axis=0), 1/np.sqrt(len(x)))
            if self.data_mean[index] is None:
                self.data_mean[index] = np.mean(x, axis=0)
            return (x - self.data_mean[index]) / self.data_std[index]

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.repeat((np.sum(np.exp(x), axis=1))[:, np.newaxis], len(x[0]), axis=1)


class Activators:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.e ** (-x))

    @staticmethod
    def sine(x):
        return np.sin(x)

    @staticmethod
    def hardlim(x):
        return (np.sign(x) + 1) / 2

    @staticmethod
    def tribas(x):
        return np.maximum(1 - np.abs(x), 0)

    @staticmethod
    def radbas(x):
        return np.exp(-(x**2))

    @staticmethod
    def sign(x):
        return np.sign(x)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def leaky_relu(x):
        x[x >= 0] = x[x >= 0]
        x[x < 0] = x[x < 0] / 10.0
        return x
