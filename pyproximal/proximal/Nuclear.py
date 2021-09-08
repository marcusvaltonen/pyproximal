import numpy as np

from pylops.optimization.sparsity import _softthreshold
from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator


class Nuclear(ProxOperator):
    r"""Nuclear norm proximal operator.

    Proximal operator of the Nuclear norm defined as
    :math:`\sigma||\mathbf{X}||_* = \sigma \sum_i \lambda_i` where
    :math:`\mathbf{X}` is a matrix of size :math:`N \times M` and
    :math:`\lambda_i=1,2, min(N, M)` are its eigenvalues.

    Parameters
    ----------
    dim : :obj:`tuple`
        Size of matrix :math:`\mathbf{X}`
    sigma : :obj:`int`, optional
        Multiplicative coefficient of Nuclear norm

    Notes
    -----
    The Nuclear norm proximal operator is defined as:

    .. math::

        prox_{\tau \sigma ||.||_*}(\mathbf{X}) =
        \mathbf{U} diag{prox_{\tau \sigma ||.||_1}(\boldsymbol\lambda)} \mathbf{V}

    where :math:`\mathbf{U}`, :math:`\boldsymbol\lambda}`, and
    :math:`\mathbf{V}` define the SVD of :math:`X`.

    """
    def __init__(self, dim, sigma=1.):
        super().__init__(None, False)
        self.dim = dim
        self.sigma = sigma

    def __call__(self, x):
        X = x.reshape(self.dim)
        eigs = np.linalg.eigvalsh(X.T @ X)
        eigs[eigs < 0] = 0 # ensure all eigenvalues at positive
        nucl = np.sum(np.sqrt(eigs))
        return self.sigma * nucl

    @_check_tau
    def prox(self, x, tau):
        X = x.reshape(self.dim)
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        Sth = _softthreshold(S, tau * self.sigma)
        X = np.dot(U * Sth, Vh)
        return X.ravel()