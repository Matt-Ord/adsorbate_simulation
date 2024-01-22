"""
This type stub file was generated by pyright.
"""

__all__ = ["cossin"]

def cossin(
    X, p=..., q=..., separate=..., swap_sign=..., compute_u=..., compute_vh=...
):  # -> tuple[tuple[Unknown, Unknown], Unknown, tuple[Unknown, Unknown]] | tuple[Unknown, NDArray[float64], Unknown]:
    """
    Compute the cosine-sine (CS) decomposition of an orthogonal/unitary matrix.

    X is an ``(m, m)`` orthogonal/unitary matrix, partitioned as the following
    where upper left block has the shape of ``(p, q)``::

                                   ┌                   ┐
                                   │ I  0  0 │ 0  0  0 │
        ┌           ┐   ┌         ┐│ 0  C  0 │ 0 -S  0 │┌         ┐*
        │ X11 │ X12 │   │ U1 │    ││ 0  0  0 │ 0  0 -I ││ V1 │    │
        │ ────┼──── │ = │────┼────││─────────┼─────────││────┼────│
        │ X21 │ X22 │   │    │ U2 ││ 0  0  0 │ I  0  0 ││    │ V2 │
        └           ┘   └         ┘│ 0  S  0 │ 0  C  0 │└         ┘
                                   │ 0  0  I │ 0  0  0 │
                                   └                   ┘

    ``U1``, ``U2``, ``V1``, ``V2`` are square orthogonal/unitary matrices of
    dimensions ``(p,p)``, ``(m-p,m-p)``, ``(q,q)``, and ``(m-q,m-q)``
    respectively, and ``C`` and ``S`` are ``(r, r)`` nonnegative diagonal
    matrices satisfying ``C^2 + S^2 = I`` where ``r = min(p, m-p, q, m-q)``.

    Moreover, the rank of the identity matrices are ``min(p, q) - r``,
    ``min(p, m - q) - r``, ``min(m - p, q) - r``, and ``min(m - p, m - q) - r``
    respectively.

    X can be supplied either by itself and block specifications p, q or its
    subblocks in an iterable from which the shapes would be derived. See the
    examples below.

    Parameters
    ----------
    X : array_like, iterable
        complex unitary or real orthogonal matrix to be decomposed, or iterable
        of subblocks ``X11``, ``X12``, ``X21``, ``X22``, when ``p``, ``q`` are
        omitted.
    p : int, optional
        Number of rows of the upper left block ``X11``, used only when X is
        given as an array.
    q : int, optional
        Number of columns of the upper left block ``X11``, used only when X is
        given as an array.
    separate : bool, optional
        if ``True``, the low level components are returned instead of the
        matrix factors, i.e. ``(u1,u2)``, ``theta``, ``(v1h,v2h)`` instead of
        ``u``, ``cs``, ``vh``.
    swap_sign : bool, optional
        if ``True``, the ``-S``, ``-I`` block will be the bottom left,
        otherwise (by default) they will be in the upper right block.
    compute_u : bool, optional
        if ``False``, ``u`` won't be computed and an empty array is returned.
    compute_vh : bool, optional
        if ``False``, ``vh`` won't be computed and an empty array is returned.

    Returns
    -------
    u : ndarray
        When ``compute_u=True``, contains the block diagonal orthogonal/unitary
        matrix consisting of the blocks ``U1`` (``p`` x ``p``) and ``U2``
        (``m-p`` x ``m-p``) orthogonal/unitary matrices. If ``separate=True``,
        this contains the tuple of ``(U1, U2)``.
    cs : ndarray
        The cosine-sine factor with the structure described above.
         If ``separate=True``, this contains the ``theta`` array containing the
         angles in radians.
    vh : ndarray
        When ``compute_vh=True`, contains the block diagonal orthogonal/unitary
        matrix consisting of the blocks ``V1H`` (``q`` x ``q``) and ``V2H``
        (``m-q`` x ``m-q``) orthogonal/unitary matrices. If ``separate=True``,
        this contains the tuple of ``(V1H, V2H)``.

    References
    ----------
    .. [1] Brian D. Sutton. Computing the complete CS decomposition. Numer.
           Algorithms, 50(1):33-65, 2009.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import cossin
    >>> from scipy.stats import unitary_group
    >>> x = unitary_group.rvs(4)
    >>> u, cs, vdh = cossin(x, p=2, q=2)
    >>> np.allclose(x, u @ cs @ vdh)
    True

    Same can be entered via subblocks without the need of ``p`` and ``q``. Also
    let's skip the computation of ``u``

    >>> ue, cs, vdh = cossin((x[:2, :2], x[:2, 2:], x[2:, :2], x[2:, 2:]),
    ...                      compute_u=False)
    >>> print(ue)
    []
    >>> np.allclose(x, u @ cs @ vdh)
    True

    """
    ...
