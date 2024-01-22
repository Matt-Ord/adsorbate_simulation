"""
This type stub file was generated by pyright.
"""

class CanonicalConstraint:
    """Canonical constraint to use with trust-constr algorithm.

    It represents the set of constraints of the form::

        f_eq(x) = 0
        f_ineq(x) <= 0

    where ``f_eq`` and ``f_ineq`` are evaluated by a single function, see
    below.

    The class is supposed to be instantiated by factory methods, which
    should prepare the parameters listed below.

    Parameters
    ----------
    n_eq, n_ineq : int
        Number of equality and inequality constraints respectively.
    fun : callable
        Function defining the constraints. The signature is
        ``fun(x) -> c_eq, c_ineq``, where ``c_eq`` is ndarray with `n_eq`
        components and ``c_ineq`` is ndarray with `n_ineq` components.
    jac : callable
        Function to evaluate the Jacobian of the constraint. The signature
        is ``jac(x) -> J_eq, J_ineq``, where ``J_eq`` and ``J_ineq`` are
        either ndarray of csr_matrix of shapes (n_eq, n) and (n_ineq, n),
        respectively.
    hess : callable
        Function to evaluate the Hessian of the constraints multiplied
        by Lagrange multipliers, that is
        ``dot(f_eq, v_eq) + dot(f_ineq, v_ineq)``. The signature is
        ``hess(x, v_eq, v_ineq) -> H``, where ``H`` has an implied
        shape (n, n) and provide a matrix-vector product operation
        ``H.dot(p)``.
    keep_feasible : ndarray, shape (n_ineq,)
        Mask indicating which inequality constraints should be kept feasible.
    """
    def __init__(self, n_eq, n_ineq, fun, jac, hess, keep_feasible) -> None: ...
    @classmethod
    def from_PreparedConstraint(cls, constraint):  # -> Self@CanonicalConstraint:
        """Create an instance from `PreparedConstrained` object."""
        ...

    @classmethod
    def empty(cls, n):  # -> Self@CanonicalConstraint:
        """Create an "empty" instance.

        This "empty" instance is required to allow working with unconstrained
        problems as if they have some constraints.
        """
        ...

    @classmethod
    def concatenate(
        cls, canonical_constraints, sparse_jacobian
    ):  # -> Self@CanonicalConstraint:
        """Concatenate multiple `CanonicalConstraint` into one.

        `sparse_jacobian` (bool) determines the Jacobian format of the
        concatenated constraint. Note that items in `canonical_constraints`
        must have their Jacobians in the same format.
        """
        ...

def initial_constraints_as_canonical(
    n, prepared_constraints, sparse_jacobian
):  # -> tuple[NDArray[Unknown] | NDArray[float64], NDArray[Unknown] | NDArray[float64], Unknown | NDArray[Unknown] | csr_matrix | NDArray[float64], Unknown | NDArray[Unknown] | csr_matrix | NDArray[float64]]:
    """Convert initial values of the constraints to the canonical format.

    The purpose to avoid one additional call to the constraints at the initial
    point. It takes saved values in `PreparedConstraint`, modififies and
    concatenates them to the canonical constraint format.
    """
    ...
