"""
This type stub file was generated by pyright.
"""

class CensoredData:
    """
    Instances of this class represent censored data.

    Instances may be passed to the ``fit`` method of continuous
    univariate SciPy distributions for maximum likelihood estimation.
    The *only* method of the univariate continuous distributions that
    understands `CensoredData` is the ``fit`` method.  An instance of
    `CensoredData` can not be passed to methods such as ``pdf`` and
    ``cdf``.

    An observation is said to be *censored* when the precise value is unknown,
    but it has a known upper and/or lower bound.  The conventional terminology
    is:

    * left-censored: an observation is below a certain value but it is
      unknown by how much.
    * right-censored: an observation is above a certain value but it is
      unknown by how much.
    * interval-censored: an observation lies somewhere on an interval between
      two values.

    Left-, right-, and interval-censored data can be represented by
    `CensoredData`.

    For convenience, the class methods ``left_censored`` and
    ``right_censored`` are provided to create a `CensoredData`
    instance from a single one-dimensional array of measurements
    and a corresponding boolean array to indicate which measurements
    are censored.  The class method ``interval_censored`` accepts two
    one-dimensional arrays that hold the lower and upper bounds of the
    intervals.

    Parameters
    ----------
    uncensored : array_like, 1D
        Uncensored observations.
    left : array_like, 1D
        Left-censored observations.
    right : array_like, 1D
        Right-censored observations.
    interval : array_like, 2D, with shape (m, 2)
        Interval-censored observations.  Each row ``interval[k, :]``
        represents the interval for the kth interval-censored observation.

    Notes
    -----
    In the input array `interval`, the lower bound of the interval may
    be ``-inf``, and the upper bound may be ``inf``, but at least one must be
    finite. When the lower bound is ``-inf``, the row represents a left-
    censored observation, and when the upper bound is ``inf``, the row
    represents a right-censored observation.  If the length of an interval
    is 0 (i.e. ``interval[k, 0] == interval[k, 1]``, the observation is
    treated as uncensored.  So one can represent all the types of censored
    and uncensored data in ``interval``, but it is generally more convenient
    to use `uncensored`, `left` and `right` for uncensored, left-censored and
    right-censored observations, respectively.

    Examples
    --------
    In the most general case, a censored data set may contain values that
    are left-censored, right-censored, interval-censored, and uncensored.
    For example, here we create a data set with five observations.  Two
    are uncensored (values 1 and 1.5), one is a left-censored observation
    of 0, one is a right-censored observation of 10 and one is
    interval-censored in the interval [2, 3].

    >>> import numpy as np
    >>> from scipy.stats import CensoredData
    >>> data = CensoredData(uncensored=[1, 1.5], left=[0], right=[10],
    ...                     interval=[[2, 3]])
    >>> print(data)
    CensoredData(5 values: 2 not censored, 1 left-censored,
    1 right-censored, 1 interval-censored)

    Equivalently,

    >>> data = CensoredData(interval=[[1, 1],
    ...                               [1.5, 1.5],
    ...                               [-np.inf, 0],
    ...                               [10, np.inf],
    ...                               [2, 3]])
    >>> print(data)
    CensoredData(5 values: 2 not censored, 1 left-censored,
    1 right-censored, 1 interval-censored)

    A common case is to have a mix of uncensored observations and censored
    observations that are all right-censored (or all left-censored). For
    example, consider an experiment in which six devices are started at
    various times and left running until they fail.  Assume that time is
    measured in hours, and the experiment is stopped after 30 hours, even
    if all the devices have not failed by that time.  We might end up with
    data such as this::

        Device  Start-time  Fail-time  Time-to-failure
           1         0         13           13
           2         2         24           22
           3         5         22           17
           4         8         23           15
           5        10        ***          >20
           6        12        ***          >18

    Two of the devices had not failed when the experiment was stopped;
    the observations of the time-to-failure for these two devices are
    right-censored.  We can represent this data with

    >>> data = CensoredData(uncensored=[13, 22, 17, 15], right=[20, 18])
    >>> print(data)
    CensoredData(6 values: 4 not censored, 2 right-censored)

    Alternatively, we can use the method `CensoredData.right_censored` to
    create a representation of this data.  The time-to-failure observations
    are put the list ``ttf``.  The ``censored`` list indicates which values
    in ``ttf`` are censored.

    >>> ttf = [13, 22, 17, 15, 20, 18]
    >>> censored = [False, False, False, False, True, True]

    Pass these lists to `CensoredData.right_censored` to create an
    instance of `CensoredData`.

    >>> data = CensoredData.right_censored(ttf, censored)
    >>> print(data)
    CensoredData(6 values: 4 not censored, 2 right-censored)

    If the input data is interval censored and already stored in two
    arrays, one holding the low end of the intervals and another
    holding the high ends, the class method ``interval_censored`` can
    be used to create the `CensoredData` instance.

    This example creates an instance with four interval-censored values.
    The intervals are [10, 11], [0.5, 1], [2, 3], and [12.5, 13.5].

    >>> a = [10, 0.5, 2, 12.5]  # Low ends of the intervals
    >>> b = [11, 1.0, 3, 13.5]  # High ends of the intervals
    >>> data = CensoredData.interval_censored(low=a, high=b)
    >>> print(data)
    CensoredData(4 values: 0 not censored, 4 interval-censored)

    Finally, we create and censor some data from the `weibull_min`
    distribution, and then fit `weibull_min` to that data. We'll assume
    that the location parameter is known to be 0.

    >>> from scipy.stats import weibull_min
    >>> rng = np.random.default_rng()

    Create the random data set.

    >>> x = weibull_min.rvs(2.5, loc=0, scale=30, size=250, random_state=rng)
    >>> x[x > 40] = 40  # Right-censor values greater or equal to 40.

    Create the `CensoredData` instance with the `right_censored` method.
    The censored values are those where the value is 40.

    >>> data = CensoredData.right_censored(x, x == 40)
    >>> print(data)
    CensoredData(250 values: 215 not censored, 35 right-censored)

    35 values have been right-censored.

    Fit `weibull_min` to the censored data.  We expect to shape and scale
    to be approximately 2.5 and 30, respectively.

    >>> weibull_min.fit(data, floc=0)
    (2.3575922823897315, 0, 30.40650074451254)

    """
    def __init__(
        self, uncensored=..., *, left=..., right=..., interval=...
    ) -> None: ...
    def __repr__(self):  # -> str:
        ...
    def __str__(self) -> str: ...
    def __sub__(self, other):  # -> CensoredData:
        ...
    def __truediv__(self, other):  # -> CensoredData:
        ...
    def __len__(self):  # -> int:
        """
        The number of values (censored and not censored).
        """
        ...

    def num_censored(self):  # -> int:
        """
        Number of censored values.
        """
        ...

    @classmethod
    def right_censored(cls, x, censored):  # -> Self@CensoredData:
        """
        Create a `CensoredData` instance of right-censored data.

        Parameters
        ----------
        x : array_like
            `x` is the array of observed data or measurements.
            `x` must be a one-dimensional sequence of finite numbers.
        censored : array_like of bool
            `censored` must be a one-dimensional sequence of boolean
            values.  If ``censored[k]`` is True, the corresponding value
            in `x` is right-censored.  That is, the value ``x[k]``
            is the lower bound of the true (but unknown) value.

        Returns
        -------
        data : `CensoredData`
            An instance of `CensoredData` that represents the
            collection of uncensored and right-censored values.

        Examples
        --------
        >>> from scipy.stats import CensoredData

        Two uncensored values (4 and 10) and two right-censored values
        (24 and 25).

        >>> data = CensoredData.right_censored([4, 10, 24, 25],
        ...                                    [False, False, True, True])
        >>> data
        CensoredData(uncensored=array([ 4., 10.]),
        left=array([], dtype=float64), right=array([24., 25.]),
        interval=array([], shape=(0, 2), dtype=float64))
        >>> print(data)
        CensoredData(4 values: 2 not censored, 2 right-censored)
        """
        ...

    @classmethod
    def left_censored(cls, x, censored):  # -> Self@CensoredData:
        """
        Create a `CensoredData` instance of left-censored data.

        Parameters
        ----------
        x : array_like
            `x` is the array of observed data or measurements.
            `x` must be a one-dimensional sequence of finite numbers.
        censored : array_like of bool
            `censored` must be a one-dimensional sequence of boolean
            values.  If ``censored[k]`` is True, the corresponding value
            in `x` is left-censored.  That is, the value ``x[k]``
            is the upper bound of the true (but unknown) value.

        Returns
        -------
        data : `CensoredData`
            An instance of `CensoredData` that represents the
            collection of uncensored and left-censored values.

        Examples
        --------
        >>> from scipy.stats import CensoredData

        Two uncensored values (0.12 and 0.033) and two left-censored values
        (both 1e-3).

        >>> data = CensoredData.left_censored([0.12, 0.033, 1e-3, 1e-3],
        ...                                   [False, False, True, True])
        >>> data
        CensoredData(uncensored=array([0.12 , 0.033]),
        left=array([0.001, 0.001]), right=array([], dtype=float64),
        interval=array([], shape=(0, 2), dtype=float64))
        >>> print(data)
        CensoredData(4 values: 2 not censored, 2 left-censored)
        """
        ...

    @classmethod
    def interval_censored(cls, low, high):  # -> Self@CensoredData:
        """
        Create a `CensoredData` instance of interval-censored data.

        This method is useful when all the data is interval-censored, and
        the low and high ends of the intervals are already stored in
        separate one-dimensional arrays.

        Parameters
        ----------
        low : array_like
            The one-dimensional array containing the low ends of the
            intervals.
        high : array_like
            The one-dimensional array containing the high ends of the
            intervals.

        Returns
        -------
        data : `CensoredData`
            An instance of `CensoredData` that represents the
            collection of censored values.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import CensoredData

        ``a`` and ``b`` are the low and high ends of a collection of
        interval-censored values.

        >>> a = [0.5, 2.0, 3.0, 5.5]
        >>> b = [1.0, 2.5, 3.5, 7.0]
        >>> data = CensoredData.interval_censored(low=a, high=b)
        >>> print(data)
        CensoredData(4 values: 0 not censored, 4 interval-censored)
        """
        ...
