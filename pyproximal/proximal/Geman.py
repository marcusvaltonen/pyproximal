import numpy as np

from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator


class Geman(ProxOperator):
    r"""Geman penalty.

    The Geman penalty (named after its inventor) is a non-convex penalty [1]_.
    The pyproximal implementation considers a generalized model where

    .. math::

        Geman_{\sigma,\gamma}(\mathbf{x}) = \sum_i \frac{\sigma |x_i|}{|x_i| + \gamma}

    where :math:`{\sigma\geq 0}`, :math:`{\gamma>0}`.

    Parameters
    ----------
    sigma : :obj:`float`
        Regularization parameter.
    gamma : :obj:`float`, optional
        Regularization parameter. Default is 1.3.

    Notes
    -----
    In order to compute the proximal operator of the Geman penalty one must find the roots
    of a cubic polynomial. Consider the one-dimensional problem

    .. math::
        prox_{\tau Geman(\cdot)}(x) = argmin_{z} Geman(z) + \frac{1}{2\tau}(x - z)^2

    and assume :math:`{x\geq 0}`. Either the minimum is obtained when :math:`x=0` or when

    .. math::
        \tau\sigma\gamma + (x-y)(x+\gamma)^2 = 0 .

    The pyproximal implementation uses the closed-form solution for a cubic polynomial to
    find the minimum.

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

    @_check_tau
    def prox_vec(self, x, tau):
        out = np.zeros_like(x)
        b = 2 * self.gamma - np.abs(x)
        c = self.gamma ** 2 - 2 * self.gamma * np.abs(x)
        d = self.gamma * self.sigma * tau - self.gamma ** 2 * np.abs(x)
        idx, loc_mins = self._find_local_minima(b, c, d)
        global_min_idx = tau * self.elementwise(loc_mins) + (loc_mins - np.abs(x[idx])) ** 2 / 2 < np.abs(x[idx]) ** 2 / 2
        idx[idx] = global_min_idx
        out[idx] = np.sign(x[idx]) * loc_mins[global_min_idx]
        return out

    @staticmethod
    def _find_local_minima(b, c, d):
        f = -(c - b ** 2.0 / 3.0) ** 3.0 / 27.0
        g = (2.0 * b ** 3.0 - 9.0 * b * c + 27.0 * d) / 27.0
        idx = g ** 2.0 / 4.0 - f <= 0
        sqrtf = np.sqrt(f[idx])
        k = np.arccos(-(g[idx] / (2 * sqrtf)))
        loc_mins = 2 * sqrtf ** (1 / 3.0) * np.cos(k / 3.0) - b[idx] / 3.0
        return idx, loc_mins


if __name__ == '__main__':
    t = np.linspace(-10, 10, 20001)
    tau = 1.0
    geman = Geman(0.9, 0.1)
    import time
    tstart = time.time()
    res1 = geman.prox_vec(t, tau)
    new_time = time.time()-tstart
    print(f'time: {new_time}')
    tstart = time.time()
    res2 = geman.prox(t, tau)
    before_time = time.time() - tstart
    print(f'time: {before_time}')

    print(f'speedup: {before_time/new_time}')

    np.testing.assert_array_almost_equal(res1, res2, 12)

    print('ok')
