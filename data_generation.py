import numpy as np
import scipy.io as sio
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression


def load_lr_data(dir, rank, comm_size, lam, bias=False):
    #data = sio.loadmat(dir)
    #fea = data['C']
    fea = np.load('lr_data.npy', allow_pickle=True, encoding='bytes')
    #label = np.squeeze(data['y'])
    label = np.load('lr_label.npy', allow_pickle=True, encoding='bytes')
    opt_model = standard_lr_solve(fea, label, lam, bias)
    [m, p] = fea.shape
    print(m, p, label.shape)
    mi = int(m / comm_size)
    local_data = fea[rank * mi: (rank + 1) * mi, :]
    local_label = label[rank * mi: (rank + 1) * mi]

    return local_data, local_label, opt_model

def standard_lr_solve(data, label, lam, bias=False):
    "use standard library to solve the lr model"
    clf = LogisticRegression(random_state=0, solver='lbfgs', tol=1e-7, C=1.0/lam, fit_intercept=bias, warm_start=True).fit(data, label)
    #clf = linear_model.SGDClassifier(loss='log', alpha=lam, learning_rate='constant',  eta0=5e-3, fit_intercept=False, max_iter=1000, tol=1e-7).fit(data, label)
    opt_model = np.squeeze(clf.coef_)
    print(clf.n_iter_)
    return opt_model


def generate_lr_data(data_size, fea_size):
    train_data = np.random.normal(0, 1, (data_size, fea_size))
    coef = np.random.normal(0, 1, (fea_size, ))
    prob = 1.0 / (1 + np.exp(- train_data @ coef))
    label = np.random.binomial(1, prob)
    label[label == 0] = -1

    return train_data, label

if __name__ == '__main__':
    data_size = 800
    fea_size = 5
    data, label = generate_lr_data(data_size, fea_size)
    np.save('lr_data.npy', np.squeeze(data))
    np.save('lr_label.npy', np.squeeze(label))