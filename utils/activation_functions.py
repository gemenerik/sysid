from numpy import exp


def radial_basis_function(x, a, c):
    v = (x - c) ** 2
    # v = np.square(w) * (x - c)**2
    return a * exp(-v)


def diff_radial_basis_function(x, a, c):
    v = (x - c) ** 2
    # v = np.square(w) * (x - c)**2)
    return -2*a*(x-c)*exp(-v)


def alt_diff_radial_basis_function(x, a, c):
    rad = radial_basis_function(x, a, c)
    return - rad


def sigmoid(x, a, c):
    # v = (x - c) ** 2
    v = x
    return 2/(1+exp(-2*v))-1


def diff_sigmoid(x, a, c):
    return 4 * 1/(1+exp(-2*x)) * (1 - 1/(1+exp(-2*x)))