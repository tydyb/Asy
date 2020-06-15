import numpy as np

class lr_model:
    'least square optimization model'

    def __init__(self, A, data_size, fea_size, b, lam=1e-3):
        self.A = A
        self.data_size = data_size
        self.fea_size = fea_size
        self.b = b
        self.lam = lam

    def compute_obj(self, x):
        resd = self.b - self.A @ x
        obj = 0.5 * np.dot(resd, resd) + 0.5 * self.lam * np.dot(x, x)
        return obj

    def compute_grad(self, x):
        resd = self.b - self.A @ x
        grad = - self.A.T @ resd + self.lam * x
        return grad

    def compute_obj_grad(self, x):
        resd = self.b - self.A @ x
        obj = 0.5 * np.dot(resd, resd) + 0.5 * self.lam * np.dot(x, x)
        grad = - self.A.T @ resd + self.lam * x

        return obj, grad