"""
This type stub file was generated by pyright.
"""

__all__ = ["Covariance"]

class Covariance:
    """
    Representation of a covariance matrix

    Calculations involving covariance matrices (e.g. data whitening,
    multivariate normal function evaluation) are often performed more
    efficiently using a decomposition of the covariance matrix instead of the
    covariance metrix itself. This class allows the user to construct an
    object representing a covariance matrix using any of several
    decompositions and perform calculations using a common interface.

    .. note::

        The `Covariance` class cannot be instantiated directly. Instead, use
        one of the factory methods (e.g. `Covariance.from_diagonal`).

    Examples
    --------
    The `Covariance` class is is used by calling one of its
    factory methods to create a `Covariance` object, then pass that
    representation of the `Covariance` matrix as a shape parameter of a
    multivariate distribution.

    For instance, the multivariate normal distribution can accept an array
    representing a covariance matrix:

    >>> from scipy import stats
    >>> import numpy as np
    >>> d = [1, 2, 3]
    >>> A = np.diag(d)  # a diagonal covariance matrix
    >>> x = [4, -2, 5]  # a point of interest
    >>> dist = stats.multivariate_normal(mean=[0, 0, 0], cov=A)
    >>> dist.pdf(x)
    4.9595685102808205e-08

    but the calculations are performed in a very generic way that does not
    take advantage of any special properties of the covariance matrix. Because
    our covariance matrix is diagonal, we can use ``Covariance.from_diagonal``
    to create an object representing the covariance matrix, and
    `multivariate_normal` can use this to compute the probability density
    function more efficiently.

    >>> cov = stats.Covariance.from_diagonal(d)
    >>> dist = stats.multivariate_normal(mean=[0, 0, 0], cov=cov)
    >>> dist.pdf(x)
    4.9595685102808205e-08

    """
    def __init__(self) -> None: ...
    @staticmethod
    def from_diagonal(diagonal):  # -> CovViaDiagonal:
        r"""
        Return a representation of a covariance matrix from its diagonal.

        Parameters
        ----------
        diagonal : array_like
            The diagonal elements of a diagonal matrix.

        Notes
        -----
        Let the diagonal elements of a diagonal covariance matrix :math:`D` be
        stored in the vector :math:`d`.

        When all elements of :math:`d` are strictly positive, whitening of a
        data point :math:`x` is performed by computing
        :math:`x \cdot d^{-1/2}`, where the inverse square root can be taken
        element-wise.
        :math:`\log\det{D}` is calculated as :math:`-2 \sum(\log{d})`,
        where the :math:`\log` operation is performed element-wise.

        This `Covariance` class supports singular covariance matrices. When
        computing ``_log_pdet``, non-positive elements of :math:`d` are
        ignored. Whitening is not well defined when the point to be whitened
        does not lie in the span of the columns of the covariance matrix. The
        convention taken here is to treat the inverse square root of
        non-positive elements of :math:`d` as zeros.

        Examples
        --------
        Prepare a symmetric positive definite covariance matrix ``A`` and a
        data point ``x``.

        >>> import numpy as np
        >>> from scipy import stats
        >>> rng = np.random.default_rng()
        >>> n = 5
        >>> A = np.diag(rng.random(n))
        >>> x = rng.random(size=n)

        Extract the diagonal from ``A`` and create the `Covariance` object.

        >>> d = np.diag(A)
        >>> cov = stats.Covariance.from_diagonal(d)

        Compare the functionality of the `Covariance` object against a
        reference implementations.

        >>> res = cov.whiten(x)
        >>> ref = np.diag(d**-0.5) @ x
        >>> np.allclose(res, ref)
        True
        >>> res = cov.log_pdet
        >>> ref = np.linalg.slogdet(A)[-1]
        >>> np.allclose(res, ref)
        True

        """
        ...

    @staticmethod
    def from_precision(precision, covariance=...):  # -> CovViaPrecision:
        r"""
        Return a representation of a covariance from its precision matrix.

        Parameters
        ----------
        precision : array_like
            The precision matrix; that is, the inverse of a square, symmetric,
            positive definite covariance matrix.
        covariance : array_like, optional
            The square, symmetric, positive definite covariance matrix. If not
            provided, this may need to be calculated (e.g. to evaluate the
            cumulative distribution function of
            `scipy.stats.multivariate_normal`) by inverting `precision`.

        Notes
        -----
        Let the covariance matrix be :math:`A`, its precision matrix be
        :math:`P = A^{-1}`, and :math:`L` be the lower Cholesky factor such
        that :math:`L L^T = P`.
        Whitening of a data point :math:`x` is performed by computing
        :math:`x^T L`. :math:`\log\det{A}` is calculated as
        :math:`-2tr(\log{L})`, where the :math:`\log` operation is performed
        element-wise.

        This `Covariance` class does not support singular covariance matrices
        because the precision matrix does not exist for a singular covariance
        matrix.

        Examples
        --------
        Prepare a symmetric positive definite precision matrix ``P`` and a
        data point ``x``. (If the precision matrix is not already available,
        consider the other factory methods of the `Covariance` class.)

        >>> import numpy as np
        >>> from scipy import stats
        >>> rng = np.random.default_rng()
        >>> n = 5
        >>> P = rng.random(size=(n, n))
        >>> P = P @ P.T  # a precision matrix must be positive definite
        >>> x = rng.random(size=n)

        Create the `Covariance` object.

        >>> cov = stats.Covariance.from_precision(P)

        Compare the functionality of the `Covariance` object against
        reference implementations.

        >>> res = cov.whiten(x)
        >>> ref = x @ np.linalg.cholesky(P)
        >>> np.allclose(res, ref)
        True
        >>> res = cov.log_pdet
        >>> ref = -np.linalg.slogdet(P)[-1]
        >>> np.allclose(res, ref)
        True

        """
        ...

    @staticmethod
    def from_cholesky(cholesky):  # -> CovViaCholesky:
        r"""
        Representation of a covariance provided via the (lower) Cholesky factor

        Parameters
        ----------
        cholesky : array_like
            The lower triangular Cholesky factor of the covariance matrix.

        Notes
        -----
        Let the covariance matrix be :math:`A`and :math:`L` be the lower
        Cholesky factor such that :math:`L L^T = A`.
        Whitening of a data point :math:`x` is performed by computing
        :math:`L^{-1} x`. :math:`\log\det{A}` is calculated as
        :math:`2tr(\log{L})`, where the :math:`\log` operation is performed
        element-wise.

        This `Covariance` class does not support singular covariance matrices
        because the Cholesky decomposition does not exist for a singular
        covariance matrix.

        Examples
        --------
        Prepare a symmetric positive definite covariance matrix ``A`` and a
        data point ``x``.

        >>> import numpy as np
        >>> from scipy import stats
        >>> rng = np.random.default_rng()
        >>> n = 5
        >>> A = rng.random(size=(n, n))
        >>> A = A @ A.T  # make the covariance symmetric positive definite
        >>> x = rng.random(size=n)

        Perform the Cholesky decomposition of ``A`` and create the
        `Covariance` object.

        >>> L = np.linalg.cholesky(A)
        >>> cov = stats.Covariance.from_cholesky(L)

        Compare the functionality of the `Covariance` object against
        reference implementation.

        >>> from scipy.linalg import solve_triangular
        >>> res = cov.whiten(x)
        >>> ref = solve_triangular(L, x, lower=True)
        >>> np.allclose(res, ref)
        True
        >>> res = cov.log_pdet
        >>> ref = np.linalg.slogdet(A)[-1]
        >>> np.allclose(res, ref)
        True

        """
        ...

    @staticmethod
    def from_eigendecomposition(eigendecomposition):  # -> CovViaEigendecomposition:
        r"""
        Representation of a covariance provided via eigendecomposition

        Parameters
        ----------
        eigendecomposition : sequence
            A sequence (nominally a tuple) containing the eigenvalue and
            eigenvector arrays as computed by `scipy.linalg.eigh` or
            `numpy.linalg.eigh`.

        Notes
        -----
        Let the covariance matrix be :math:`A`, let :math:`V` be matrix of
        eigenvectors, and let :math:`W` be the diagonal matrix of eigenvalues
        such that `V W V^T = A`.

        When all of the eigenvalues are strictly positive, whitening of a
        data point :math:`x` is performed by computing
        :math:`x^T (V W^{-1/2})`, where the inverse square root can be taken
        element-wise.
        :math:`\log\det{A}` is calculated as  :math:`tr(\log{W})`,
        where the :math:`\log` operation is performed element-wise.

        This `Covariance` class supports singular covariance matrices. When
        computing ``_log_pdet``, non-positive eigenvalues are ignored.
        Whitening is not well defined when the point to be whitened
        does not lie in the span of the columns of the covariance matrix. The
        convention taken here is to treat the inverse square root of
        non-positive eigenvalues as zeros.

        Examples
        --------
        Prepare a symmetric positive definite covariance matrix ``A`` and a
        data point ``x``.

        >>> import numpy as np
        >>> from scipy import stats
        >>> rng = np.random.default_rng()
        >>> n = 5
        >>> A = rng.random(size=(n, n))
        >>> A = A @ A.T  # make the covariance symmetric positive definite
        >>> x = rng.random(size=n)

        Perform the eigendecomposition of ``A`` and create the `Covariance`
        object.

        >>> w, v = np.linalg.eigh(A)
        >>> cov = stats.Covariance.from_eigendecomposition((w, v))

        Compare the functionality of the `Covariance` object against
        reference implementations.

        >>> res = cov.whiten(x)
        >>> ref = x @ (v @ np.diag(w**-0.5))
        >>> np.allclose(res, ref)
        True
        >>> res = cov.log_pdet
        >>> ref = np.linalg.slogdet(A)[-1]
        >>> np.allclose(res, ref)
        True

        """
        ...

    def whiten(self, x):
        """
        Perform a whitening transformation on data.

        "Whitening" ("white" as in "white noise", in which each frequency has
        equal magnitude) transforms a set of random variables into a new set of
        random variables with unit-diagonal covariance. When a whitening
        transform is applied to a sample of points distributed according to
        a multivariate normal distribution with zero mean, the covariance of
        the transformed sample is approximately the identity matrix.

        Parameters
        ----------
        x : array_like
            An array of points. The last dimension must correspond with the
            dimensionality of the space, i.e., the number of columns in the
            covariance matrix.

        Returns
        -------
        x_ : array_like
            The transformed array of points.

        References
        ----------
        .. [1] "Whitening Transformation". Wikipedia.
               https://en.wikipedia.org/wiki/Whitening_transformation
        .. [2] Novak, Lukas, and Miroslav Vorechovsky. "Generalization of
               coloring linear transformation". Transactions of VSB 18.2
               (2018): 31-35. :doi:`10.31490/tces-2018-0013`

        Examples
        --------
        >>> import numpy as np
        >>> from scipy import stats
        >>> rng = np.random.default_rng()
        >>> n = 3
        >>> A = rng.random(size=(n, n))
        >>> cov_array = A @ A.T  # make matrix symmetric positive definite
        >>> precision = np.linalg.inv(cov_array)
        >>> cov_object = stats.Covariance.from_precision(precision)
        >>> x = rng.multivariate_normal(np.zeros(n), cov_array, size=(10000))
        >>> x_ = cov_object.whiten(x)
        >>> np.cov(x_, rowvar=False)  # near-identity covariance
        array([[0.97862122, 0.00893147, 0.02430451],
               [0.00893147, 0.96719062, 0.02201312],
               [0.02430451, 0.02201312, 0.99206881]])

        """
        ...

    def colorize(self, x):
        """
        Perform a colorizing transformation on data.

        "Colorizing" ("color" as in "colored noise", in which different
        frequencies may have different magnitudes) transforms a set of
        uncorrelated random variables into a new set of random variables with
        the desired covariance. When a coloring transform is applied to a
        sample of points distributed according to a multivariate normal
        distribution with identity covariance and zero mean, the covariance of
        the transformed sample is approximately the covariance matrix used
        in the coloring transform.

        Parameters
        ----------
        x : array_like
            An array of points. The last dimension must correspond with the
            dimensionality of the space, i.e., the number of columns in the
            covariance matrix.

        Returns
        -------
        x_ : array_like
            The transformed array of points.

        References
        ----------
        .. [1] "Whitening Transformation". Wikipedia.
               https://en.wikipedia.org/wiki/Whitening_transformation
        .. [2] Novak, Lukas, and Miroslav Vorechovsky. "Generalization of
               coloring linear transformation". Transactions of VSB 18.2
               (2018): 31-35. :doi:`10.31490/tces-2018-0013`

        Examples
        --------
        >>> import numpy as np
        >>> from scipy import stats
        >>> rng = np.random.default_rng(1638083107694713882823079058616272161)
        >>> n = 3
        >>> A = rng.random(size=(n, n))
        >>> cov_array = A @ A.T  # make matrix symmetric positive definite
        >>> cholesky = np.linalg.cholesky(cov_array)
        >>> cov_object = stats.Covariance.from_cholesky(cholesky)
        >>> x = rng.multivariate_normal(np.zeros(n), np.eye(n), size=(10000))
        >>> x_ = cov_object.colorize(x)
        >>> cov_data = np.cov(x_, rowvar=False)
        >>> np.allclose(cov_data, cov_array, rtol=3e-2)
        True
        """
        ...

    @property
    def log_pdet(self):  # -> ndarray[Any, dtype[Any]]:
        """
        Log of the pseudo-determinant of the covariance matrix
        """
        ...

    @property
    def rank(self):  # -> ndarray[Any, dtype[Any]]:
        """
        Rank of the covariance matrix
        """
        ...

    @property
    def covariance(self):
        """
        Explicit representation of the covariance matrix
        """
        ...

    @property
    def shape(self):
        """
        Shape of the covariance array
        """
        ...

class CovViaPrecision(Covariance):
    def __init__(self, precision, covariance=...) -> None: ...

class CovViaDiagonal(Covariance):
    def __init__(self, diagonal) -> None: ...

class CovViaCholesky(Covariance):
    def __init__(self, cholesky) -> None: ...

class CovViaEigendecomposition(Covariance):
    def __init__(self, eigendecomposition) -> None: ...

class CovViaPSD(Covariance):
    """
    Representation of a covariance provided via an instance of _PSD
    """
    def __init__(self, psd) -> None: ...
