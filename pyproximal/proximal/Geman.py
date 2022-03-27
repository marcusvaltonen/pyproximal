import numpy as np

from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator


class Geman(ProxOperator):
    r"""Geman penalty.

    Parameters
    ----------
    sigma : :obj:`float`
        Regularization parameter.
    gamma : :obj:`float`, optional
        Regularization parameter. Default is 1.3.

    Notes
    -----
    The Geman penalty (named after its inventor) is a non-convex penalty [1]_.
    The pyproximal implementation considers a generalized model where

    .. math::

        Geman_{\sigma,\gamma}(\mathbf{x}) = \sum_i \frac{\sigma |x_i|}{|x_i| + \gamma}

    where :math:`{\sigma>0}`, :math:`{\gamma>0}`.
    .. [1] Geman and Yang "Nonlinear image recovery with half-quadratic regularization",
        IEEE Transactions on Image Processing, 4(7):932 – 946, 1995.

    """

    def __init__(self, sigma, gamma=1.3):
        super().__init__(None, False)
        if sigma < 0:
            raise ValueError('Variable "sigma" must be positive.')
        if gamma <= 0:
            raise ValueError('Variable "gamma" must be strictly positive.')
        self.sigma = sigma
        self.gamma = gamma

    def __call__(self, x):
        return np.sum(self.elementwise(x))

    def elementwise(self, x):
        return self.sigma * np.abs(x) / (np.abs(x) + self.gamma)

    @_check_tau
    def prox(self, x, tau):
        out = np.zeros_like(x)
        for i, y in enumerate(x):
            coeffs = [
                1,
                2 * self.gamma - np.abs(y),
                self.gamma ** 2 - 2 * self.gamma * np.abs(y),
                self.gamma * self.sigma * tau - self.gamma ** 2 * np.abs(y)
            ]
            r = np.append(0, np.roots(coeffs))
            r = np.real(r[np.logical_and(np.isreal(r), np.real(r) >= 0)])
            val = tau * self.elementwise(r) + (r - np.abs(y)) ** 2 / 2
            idx = np.argmin(val)
            out[i] = np.sign(y) * r[idx]
        return out
