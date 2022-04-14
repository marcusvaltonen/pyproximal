import numpy as np

from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator


class QuadraticEnvelopeCard(ProxOperator):
    r"""Quadratic envelope of the :math:`\ell_0`-penalty.

    The :math:`\ell_0`-penalty is also known as the *cardinality function*, and the
    quadratic envelope :math:`\mathcal{Q}(\mu\|\cdot\|_0)` of it is defined as

    .. math::

        \mathcal{Q}(\mu\|\cdot\|_0)(x) = \sum_i \left(\mu - \frac{1}{2}\max(0, \sqrt{2\mu} - |x_i|)^2\right)

    where :math:`\mu \geq 0`.

    Parameters
    ----------
    mu : :obj:`float`
        Threshold parameter.

    Notes
    -----
    The terminology *quadratic envelope* was coined in [1]_, however, the rationale has
    been used earlier, e.g. in [2]_. In a general setting, the quadratic envelope
    :math:`\mathcal{Q}(f)(x)` is defined such that

    .. math::

        \left(f(x) + \frac{1}{2}\|x-y\|_2^2\right)^{**} = \mathcal{Q}(f)(x) + \frac{1}{2}\|x-y\|_2^2

    where :math:`g^{**}` denotes the bi-conjugate of :math:`g`, which is the l.s.c.
    convex envelope of :math:`g`.

    There is no closed-form expression for :math:`\mathcal{Q}(f)(x)` given an arbitrary
    function :math:`f`. However, for certain special cases, such as in the case of the
    cardinality function, such expressions do exist.

    The proximal operator is given by

    .. math::

        \prox_{\tau\mathcal{Q}(\mu\|\cdot\|_0)}(x) =
        \begin{cases}
        x_i, & |x_i| \geq \sqrt{2 \mu} \\
        \frac{x_i-\tau\sqrt{2\mu}\sgn(x_i)}{1-\tau}, & \tau\sqrt{2\mu} < |x_i| < \sqrt{2 \mu} \\
        0, & |x_i| \leq \tau\sqrt{2 \mu}
        \end{cases}

    By inspecting the structure of the proximal operator it is clear that large values
    are unaffected, whereas smaller ones are penalized partially or completely. Such
    properties are desirable to counter the effect of *shrinking bias* observed with
    e.g. the :math:`\ell_1`-penalty. Note that in the limit :math:`\tau=1` this becomes
    the hard thresholding with threshold :math:`\sqrt{2\mu}`. It should also be noted
    that this proximal operator is identical to the Minimax Concave Penalty (MCP)
    proposed in [3]_.

    .. [1] Carlsson, M. "On Convex Envelopes and Regularization of Non-convex
        Functionals Without Moving Global Minima", In Journal of Optimization Theory
        and Applications, 183:66–84, 2019.
    .. [2] Larsson, V. and Olsson, C. "Convex Low Rank Approximation", In International
        Journal of Computer Vision (IJCV), 120:194–214, 2016.
    .. [3] Zhang et al. "Nearly unbiased variable selection under minimax concave
        penalty", In the Annals of Statistics, 38(2):894–942, 2010.

    """

    def __init__(self, mu):
        super().__init__(None, False)
        self.mu = mu

    def __call__(self, x):
        return np.sum(self.elementwise(x))

    def elementwise(self, x):
        return self.mu - 0.5 * np.maximum(0, np.sqrt(2 * self.mu) - np.abs(x)) ** 2

    @_check_tau
    def prox(self, x, tau):
        r = np.abs(x)
        idx = r < np.sqrt(2 * self.mu)
        if tau >= 1:
            r[idx] = 0
        else:
            r[idx] = np.maximum(0, (r[idx] - tau * np.sqrt(2 * self.mu)) / (1 - tau))
        return r * np.sign(x)


class QuadraticEnvelopeCardIndicator(ProxOperator):
    r"""Quadratic envelope of the indicator function of :math:`\ell_0`-penalty.

    The :math:`\ell_0`-penalty is also known as the *cardinality function*, and the
    indicator function :math:`\mathcal{I}_{r_0}` is defined as

    .. math::

        \mathcal{I}_{r_0}(\mathbf{x}) =
        \begin{cases}
        0, & \mathbf{x}\leq r_0 \\
        \infty, & \text{otherwise}
        \end{cases}

    Let :math:`\tilde{\mathbf{x}}` denote the vector :math:`\mathbf{x}` resorted such that the
    sequence :math:`(\tilde{x}_i)` is non-increasing. The quadratic envelope
    :math:`\mathcal{Q}(\mathcal{I}_{r_0})` can then be written as

    .. math::

        \mathcal{Q}(\mathcal{I}_{r_0})(x) =
        \frac{1}{2k^*}\left(\sum_{i>r_0-k^*}|\tilde{x}_i|\right)^2
        - \frac{1}{2}\left(\sum_{i>r_0-k^*}|\tilde{x}_i|^2

    where :math:`r_0 \geq 0` and :math:`k^* \leq r_0`, see [3]_ for details. There are
    other, equivalent ways, of expressing this penalty, see e.g. [1]_ and [2]_.

    Parameters
    ----------
    r0 : :obj:`int`
        Threshold parameter.

    Notes
    -----
    The terminology *quadratic envelope* was coined in [1]_, however, the rationale has
    been used earlier, e.g. in [2]_. In a general setting, the quadratic envelope
    :math:`\mathcal{Q}(f)(x)` is defined such that

    .. math::

        \left(f(x) + \frac{1}{2}\|x-y\|_2^2\right)^{**} = \mathcal{Q}(f)(x) + \frac{1}{2}\|x-y\|_2^2

    where :math:`g^{**}` denotes the bi-conjugate of :math:`g`, which is the l.s.c.
    convex envelope of :math:`g`.

    There is no closed-form expression for :math:`\mathcal{Q}(f)(x)` given an arbitrary
    function :math:`f`. However, for certain special cases, such as in the case of the
    indicator function of the cardinality function, such expressions do exist.

    The proximal operator is given by (FIX THIS)

    .. math::

        \prox_{\tau\mathcal{Q}(\mu\|\cdot\|_0)}(x) =
        \begin{cases}
        x_i, & |x_i| \geq \sqrt{2 \mu} \\
        \frac{x_i-\tau\sqrt{2\mu}\sgn(x_i)}{1-\tau}, & \tau\sqrt{2\mu} < |x_i| < \sqrt{2 \mu} \\
        0, & |x_i| \leq \tau\sqrt{2 \mu}
        \end{cases}

    Note that this is a non-separable penalty.

    .. [1] Carlsson, M. "On Convex Envelopes and Regularization of Non-convex
        Functionals Without Moving Global Minima", In Journal of Optimization Theory
        and Applications, 183:66–84, 2019.
    .. [2] Larsson, V. and Olsson, C. "Convex Low Rank Approximation", In International
        Journal of Computer Vision (IJCV), 120:194–214, 2016.
    .. [3] Andersson et al. "Convex envelopes for fixed rank approximation", In
        Optimization Letters, 11:1783–1795, 2017.

    """

    def __init__(self, r0):
        super().__init__(None, False)
        self.r0 = r0

    def __call__(self, x):
        # TODO: Write a hankel recovery tutorial.
        if x.size <= self.r0 or np.count_nonzero(x) <= self.r0:
            return 0
        xs = np.sort(np.abs(x))[::-1]
        sums = np.cumsum(xs[::-1])
        sums = sums[-self.r0:] / np.arange(1, self.r0 + 1)
        tmp = np.diff(sums) > 0
        k_star = np.argmax(tmp)
        if k_star == 0 and not tmp[k_star]:
            k_star = self.r0 - 1
        return 0.5 * ((k_star + 1) * sums[k_star] ** 2 - np.sum(xs[self.r0-k_star-1:] ** 2))

    @_check_tau
    def prox(self, x, tau):
        #TODO: Implement this
        """
function [ X, zk, yk, Rr0 ] = prox_r0_rank( X0, r0, rho )
% Solves the problem
%   min_X R_r0(X) + rho * |X-X0|^2

[U,S,V] = svd(X,'econ');
yk = diag(S);
n = length(yk);
[p,ind] = sort([yk(1:r0);rho*yk(r0+1:end)]);

a = (n-r0)/(rho-1);
b = rho/(rho-1) * sum(yk(r0+1:end));

% base case
zk = yk;
zk(r0+1:end) = (1+rho)*yk(r0+1:end);

for k = 1:length(p)-1
    % interval [p(k) p(k+1)]

    if ind(k) <= r0
        a = a + rho/(rho-1);
        b = b + rho/(rho-1) * yk(ind(k));
    end

    if ind(k) > r0
        a = a - 1/(rho-1);
        b = b - rho/(rho-1) * yk(ind(k));
    end

    s = b / a;

    if p(k) <= s && s <= p(k+1)
        zk = max(s, yk);
        zk(r0+1:n) = min(s, rho*yk(r0+1:n));
        break;
    end
end

c = @(s) sum(-rho/(rho-1)*max(0,s-yk(1:r0)).^2) + ...
    sum(-1/(rho-1)*max(0,(rho+1)*yk(r0+1:n)-s).^2+rho*yk(r0+1:n).^2);

Rr0 = c(s);

Z = U*diag(zk)*V';
"""
        rho = 1 / tau
        yk = np.sort(np.abs(x))[::-1]
        n = yk.size
        ind = np.argsort(np.concatenate((yk[:self.r0], rho * yk[self.r0:])))
        p = yk[ind]

        a = (n-self.r0)/(rho-1)
        b = rho/(rho-1) * np.sum(yk[self.r0:])

        # base case
        zk = yk.copy()
        zk[self.r0:] = rho * yk[self.r0:]

        for k in range(p.size - 1):
            # interval [p(k) p(k+1)]

            if ind[k] < self.r0:
                a = a + rho/(rho-1)
                b = b + rho/(rho-1) * yk[ind[k]]

            if ind[k] >= self.r0:
                a = a - 1/(rho-1)
                b = b - rho/(rho-1) * yk[ind[k]]

            s = b / a

            if p[k] <= s <= p[k + 1]:
                zk = np.maximum(s, yk)
                zk[self.r0:] = np.minimum(s, rho * yk[self.r0:])
                break

        c = lambda s: np.sum(-rho/(rho-1)*np.maximum(0,s-yk[:self.r0]) ** 2) + \
            np.sum(-1/(rho-1)*np.maximum(0,rho * yk[self.r0:]-s)**2+rho*yk[self.r0:]**2)

        Rr0 = c(s)

        return zk, Rr0


if __name__ == '__main__':
    penalty = QuadraticEnvelopeCardIndicator(4)
    for i in range(1):
        x = np.random.randn(i + 9)
        print(f'penalty={penalty(x)}')
        print(f'penalty={penalty.prox(x, 0.5)}')

