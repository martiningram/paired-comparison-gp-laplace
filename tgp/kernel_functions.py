import numpy as np


def compute_weighted_square_distances(x1, x2, rho):

    z1 = x1 / np.expand_dims(rho, axis=0)
    z2 = x2 / np.expand_dims(rho, axis=0)

    # Matrix part
    cross_contrib = -2 * np.dot(z1, z2.T)

    # Other bits
    z1_sq = np.sum(z1**2, axis=1)
    z2_sq = np.sum(z2**2, axis=1)

    # Sum it all up
    combined = (np.expand_dims(z1_sq, axis=1) + cross_contrib +
                np.expand_dims(z2_sq, axis=0))

    return combined


def ard_rbf_kernel_efficient(x1, x2, alpha, rho, jitter=1e-5):

    combined = compute_weighted_square_distances(x1, x2, rho)
    kernel = alpha**2 * np.exp(-0.5 * combined)
    kernel = add_jitter(kernel, jitter)

    return kernel


def add_jitter(kernel, jitter=1e-5):

    # Add the jitter
    diag_indices = np.diag_indices(np.min(kernel.shape[:2]))
    to_add = np.zeros_like(kernel)
    to_add[diag_indices] += jitter
    kernel = kernel + to_add

    return kernel


def matern_kernel_32(x1, x2, alpha, rho, jitter=1e-5):

    r_sq = compute_weighted_square_distances(x1, x2, rho)
    r = np.sqrt(r_sq)

    kernel = alpha ** 2 * (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
    kernel = add_jitter(kernel, jitter)

    return kernel


def matern_kernel_12(x1, x2, alpha, rho, jitter=1e-5):

    r_sq = compute_weighted_square_distances(x1, x2, rho)
    r = np.sqrt(r_sq)

    kernel = alpha**2 * np.exp(-r)
    kernel = add_jitter(kernel, jitter)

    return kernel


def brownian_kernel_1d(x1, x2, alpha, jitter=1e-5):

    assert(x1.shape[1] == 1 and x2.shape[1] == 1)

    variance = alpha ** 2

    kernel = variance * np.where(np.sign(x1) == np.sign(x2.T),
                                 np.fmin(np.abs(x1), np.abs(x2.T)), 0.)

    kernel = add_jitter(kernel, jitter)

    return kernel


def mlp_kernel(x1, x2, variance, weight_variance, bias_variance, jitter=1e-5):

    four_over_tau = 2. / np.pi

    def comp_prod(x1, x2=None):

        if x2 is None:
            return ((np.square(x1) * weight_variance).sum(axis=1) +
                    bias_variance)
        else:
            return (x1 * weight_variance).dot(x2.T) + bias_variance

    x1_denom = np.sqrt(comp_prod(x1) + 1.)
    x2_denom = np.sqrt(comp_prod(x2) + 1.)
    xtx = comp_prod(x1, x2) / x1_denom[:, None] / x2_denom[None, :]
    kern = variance * four_over_tau * np.arcsin(xtx)
    kern = add_jitter(kern, jitter)

    return kern


def rq_kernel(x1, x2, variance, lscales, alpha, jitter=1e-5):

    dists = compute_weighted_square_distances(x1, x2, lscales)

    # Divide by alpha
    divided = dists / alpha
    result = (1 + divided)**(-alpha)
    result = variance * result
    result = add_jitter(result, jitter)

    return result
