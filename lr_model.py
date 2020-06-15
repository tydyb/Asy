import numpy as np
from sklearn.linear_model import LogisticRegression

class lr_model:
    'logistic regression optimization model'

    def __init__(self, data, data_size, fea_size, label, lam=1e-3, bias=False):
        #self.data = data
        self.data_size = data_size
        self.fea_size = fea_size
        self.label = label
        self.lam = lam
        self.bias = bias
        if bias == True:
            self.data = np.zeros([self.data_size, self.fea_size + 1])
            self.data[:, 0:self.fea_size] = data
        else:
            self.data = data

    def compute_obj(self, x):
        expo = np.exp(- (self.data @ x) * self.label)
        obj = np.sum(np.log(1 + expo), axis=0) + 0.5 * self.lam * np.dot(x, x)

        return obj

    def compute_grad(self, x):
        x_dim = x.shape[0]
        expo = np.exp(- (self.data @ x) * self.label)
        grad = np.sum(np.tile(-expo / (1 + expo) * self.label, (x_dim, 1)) * self.data.T, axis=1) + self.lam * x

        return grad

    def compute_obj_grad(self, x):
        x_dim = x.shape[0]
        expo = np.exp(- (self.data @ x) * self.label)
        obj = np.sum(np.log(1 + expo), axis=0) + 0.5 * self.lam * np.dot(x, x)
        grad = np.sum(np.tile(-expo / (1 + expo) * self.label, (x_dim, 1)) * self.data.T, axis=1) + self.lam * x

        return obj, grad

    def generate_lr_data(self):
        train_data = np.random.random_sample((self.data_size, self.fea_size))
        if self.bias == True:
            train_data[:, -1] = 1

        coef = np.random.normal(0, 1, (self.fea_size,))
        prob = 1.0 / (1 + np.exp(- train_data @ coef))
        label = np.random.binomial(1, prob)
        label[label == 0] = -1
        self.gene_data = train_data
        self.gene_label = label

        return train_data, label

