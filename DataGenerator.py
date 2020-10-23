import numpy as np


def generate_data(n=100, seed=0):
    """
    Generate data (X,y)
    Args:
        n(int): number of samples
        seed: random seed used
    Return:
        X(float): [n,10].
        y(float): n dimensional array.
    Taken from https://github.com/Jianbo-Lab/CCM
    See:
    http://papers.nips.cc/paper/7270-kernel-feature-selection-via-conditional-covariance-minimization.pdf
    for details.
    """
    np.random.seed(seed)
    X = np.random.randn(n, 10)
    y = np.zeros(n)
    splits = np.linspace(0, n, num=8 + 1, dtype=int)
    signals = [[1, 1, 1], [-1, -1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [1, -1, 1]]
    for i in range(8):
        X[splits[i]:splits[i + 1], :3] += np.array([signals[i]])
        y[splits[i]:splits[i + 1]] = i // 2
    perm_inds = np.random.permutation(n)
    X, y = X[perm_inds], y[perm_inds]
    return X, y


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])
