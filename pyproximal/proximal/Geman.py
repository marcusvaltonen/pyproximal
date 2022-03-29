import numpy as np

from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator

import math


# Helper function to return float value of f.
def findF(a, b, c):
    return ((3.0 * c / a) - ((b ** 2.0) / (a ** 2.0))) / 3.0


# Helper function to return float value of g.
def findG(a, b, c, d):
    return (((2.0 * (b ** 3.0)) / (a ** 3.0)) - ((9.0 * b * c) / (a ** 2.0)) + (27.0 * d / a)) / 27.0


# Helper function to return float value of h.
def findH(g, f):
    return ((g ** 2.0) / 4.0 + (f ** 3.0) / 27.0)


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
    def prox_vec_illustrate(self, x, tau):
        out = np.zeros_like(x)
        tmp = []
        import matplotlib.pyplot as plt

        figure, ax = plt.subplots(1, 2, figsize=(4, 5))

        plt.ion()
        plot1, = ax[0].plot(t, t)
        plot2, = ax[0].plot(np.zeros(3), np.zeros(3), 'ro')

        plot3, = ax[1].plot(t, t)
        plot4, = ax[1].plot(np.zeros(3), np.zeros(3), 'ro')
        ax[1].set_ylim(0, 25)

        for i, y in enumerate(x):
            coeffs = [
                1,
                2 * self.gamma - np.abs(y),
                self.gamma ** 2 - 2 * self.gamma * np.abs(y),
                self.gamma * self.sigma * tau - self.gamma ** 2 * np.abs(y)
            ]
            #r = np.append(0, np.roots(coeffs))

            a, b, c, d = coeffs
            r = self.solve(a, b, c, d)
            r = np.real(r[np.logical_and(np.isreal(r), np.real(r) >= 0)])

            #print
            ax[0].set_title(f'x = {y}')
            plot1.set_ydata(np.polyval(coeffs, t))
            plot2.set_xdata(r)
            plot2.set_ydata(0*r)
            plot3.set_ydata(tau*self.elementwise(t)+(np.abs(y)-t)**2 / 2)
            plot4.set_xdata(r)
            plot4.set_ydata(tau*self.elementwise(r)+(np.abs(y)-r)**2 / 2)
            figure.canvas.draw()
            figure.canvas.flush_events()
            plt.pause(0.1)
            r = np.append(0, r)

            val = tau * self.elementwise(r) + (r - np.abs(y)) ** 2 / 2
            idx = np.argmin(val)
            tmp.append(idx)
            out[i] = np.sign(y) * r[idx]
        print(f'unique: {np.unique(tmp)}')
        print('??')
        return out

    @_check_tau
    def prox_vec_illustrate(self, x, tau):
        out = np.zeros_like(x)
        tmp = []
        import matplotlib.pyplot as plt

        figure, ax = plt.subplots(1, 2, figsize=(4, 5))

        plt.ion()
        plot1, = ax[0].plot(t, t)
        plot2, = ax[0].plot(np.zeros(3), np.zeros(3), 'ro')

        plot3, = ax[1].plot(t, t)
        plot4, = ax[1].plot(np.zeros(3), np.zeros(3), 'ro')
        ax[1].set_ylim(0, 25)

        for i, y in enumerate(x):
            coeffs = [
                1,
                2 * self.gamma - np.abs(y),
                self.gamma ** 2 - 2 * self.gamma * np.abs(y),
                self.gamma * self.sigma * tau - self.gamma ** 2 * np.abs(y)
            ]
            #r = np.append(0, np.roots(coeffs))

            a, b, c, d = coeffs
            r = self.solve(a, b, c, d)
            r = np.real(r[np.logical_and(np.isreal(r), np.real(r) >= 0)])

            #print
            ax[0].set_title(f'x = {y}')
            plot1.set_ydata(np.polyval(coeffs, t))
            plot2.set_xdata(r)
            plot2.set_ydata(0*r)
            plot3.set_ydata(tau*self.elementwise(t)+(np.abs(y)-t)**2 / 2)
            plot4.set_xdata(r)
            plot4.set_ydata(tau*self.elementwise(r)+(np.abs(y)-r)**2 / 2)
            figure.canvas.draw()
            figure.canvas.flush_events()
            plt.pause(0.1)
            r = np.append(0, r)

            val = tau * self.elementwise(r) + (r - np.abs(y)) ** 2 / 2
            idx = np.argmin(val)
            tmp.append(idx)
            out[i] = np.sign(y) * r[idx]
        print(f'unique: {np.unique(tmp)}')
        print('??')
        return out

    @_check_tau
    def prox_cubic(self, x, tau):
        out = np.zeros_like(x)
        for i, y in enumerate(x):
            b = 2 * self.gamma - np.abs(y)
            c = self.gamma ** 2 - 2 * self.gamma * np.abs(y)
            d = self.gamma * self.sigma * tau - self.gamma ** 2 * np.abs(y)
            loc_min = self.find_local_minimum(b, c, d)
            if loc_min and tau * self.elementwise(loc_min) + (loc_min - np.abs(y)) ** 2 / 2 < np.abs(y) ** 2 / 2:
                out[i] = np.sign(y) * loc_min

        return out

    @_check_tau
    def prox_vec(self, x, tau):
        out = np.zeros_like(x)
        b = 2 * self.gamma - np.abs(x)
        c = self.gamma ** 2 - 2 * self.gamma * np.abs(x)
        d = self.gamma * self.sigma * tau - self.gamma ** 2 * np.abs(x)
        idx, loc_mins = self.find_local_minima(b, c, d)
        global_min_idx = tau * self.elementwise(loc_mins) + (loc_mins - np.abs(x[idx])) ** 2 / 2 < np.abs(x[idx]) ** 2 / 2
        idx[idx] = global_min_idx
        out[idx] = np.sign(x[idx]) * loc_mins[global_min_idx]
        return out

    @staticmethod
    def find_local_minimum(b, c, d):
        f = -(c - b ** 2.0 / 3.0) ** 3.0 / 27.0
        g = (2.0 * b ** 3.0 - 9.0 * b * c + 27.0 * d) / 27.0
        h = g ** 2.0 / 4.0 - f
        if h <= 0:
            sqrtf = math.sqrt(f)
            k = math.acos(-(g / (2 * sqrtf)))
            return 2 * sqrtf ** (1 / 3.0) * math.cos(k / 3.0) - b / 3.0
        return None

    @staticmethod
    def find_local_minima(b, c, d):
        f = -(c - b ** 2.0 / 3.0) ** 3.0 / 27.0
        g = (2.0 * b ** 3.0 - 9.0 * b * c + 27.0 * d) / 27.0
        idx = g ** 2.0 / 4.0 - f <= 0
        sqrtf = np.sqrt(f[idx])
        k = np.arccos(-(g[idx] / (2 * sqrtf)))
        loc_mins = 2 * sqrtf ** (1 / 3.0) * np.cos(k / 3.0) - b[idx] / 3.0
        return idx, loc_mins


if __name__ == '__main__':
    t = np.linspace(-10, 10, 2001)
    tau = 1.0
    geman = Geman(0.9, 0.1)
    np.testing.assert_array_almost_equal(geman.prox_cubic(t, tau), geman.prox(t, tau), 12)
    np.testing.assert_array_almost_equal(geman.prox_vec(t, tau), geman.prox(t, tau), 12)

    print('ok')
