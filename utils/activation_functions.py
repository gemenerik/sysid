from numpy import exp


def radial_basis_function(x, a, c):
    v = (x - c) ** 2
    # v = np.square(w) * (x - c)**2
    return a * exp(-v)


def diff_radial_basis_function(x, a, c):
    v = (x - c) ** 2
    # v = np.square(w) * (x - c)**2)
    return -2*a*(x-c)*exp(-v)


def sigmoid(x, a, c):
    v = (x - c) ** 2
    return 2/(1+exp(-2*v))-1


def diff_sigmoid(x, a, c):
    # v = (x - c) ** 2
    # return 8*v*np.exp(2*np.square(v))
    return sigmoid(x, a, c) * (1 - sigmoid(x, a, c))