"""
This type stub file was generated by pyright.
"""

from typing import Any

"""
Fundamental Physical Constants
------------------------------

These constants are taken from CODATA Recommended Values of the Fundamental
Physical Constants 2018.

Object
------
physical_constants : dict
    A dictionary containing physical constants. Keys are the names of physical
    constants, values are tuples (value, units, precision).

Functions
---------
value(key):
    Returns the value of the physical constant(key).
unit(key):
    Returns the units of the physical constant(key).
precision(key):
    Returns the relative precision of the physical constant(key).
find(sub):
    Prints or returns list of keys containing the string sub, default is all.

Source
------
The values of the constants provided at this site are recommended for
international use by CODATA and are the latest available. Termed the "2018
CODATA recommended values," they are generally recognized worldwide for use in
all fields of science and technology. The values became available on 20 May
2019 and replaced the 2014 CODATA set. Also available is an introduction to the
constants for non-experts at

https://physics.nist.gov/cuu/Constants/introduction.html

References
----------
Theoretical and experimental publications relevant to the fundamental constants
and closely related precision measurements published since the mid 1980s, but
also including many older papers of particular interest, some of which date
back to the 1800s. To search the bibliography, visit

https://physics.nist.gov/cuu/Constants/

"""
__all__ = [
    "physical_constants",
    "value",
    "unit",
    "precision",
    "find",
    "ConstantWarning",
]
txt2002 = ...
txt2006 = ...
txt2010 = ...
txt2014 = ...
txt2018 = ...
physical_constants: dict[str, tuple[float, str, float]] = ...

def parse_constants_2002to2014(d: str) -> dict[str, tuple[float, str, float]]: ...
def parse_constants_2018toXXXX(d: str) -> dict[str, tuple[float, str, float]]: ...

_physical_constants_2002 = ...
_physical_constants_2006 = ...
_physical_constants_2010 = ...
_physical_constants_2014 = ...
_physical_constants_2018 = ...
_current_constants = ...
_current_codata = ...
_obsolete_constants = ...
_aliases = ...

class ConstantWarning(DeprecationWarning):
    """Accessing a constant no longer in current CODATA data set"""

    ...

def value(key: str) -> float:
    """
    Value in physical_constants indexed by key

    Parameters
    ----------
    key : Python string
        Key in dictionary `physical_constants`

    Returns
    -------
    value : float
        Value in `physical_constants` corresponding to `key`

    Examples
    --------
    >>> from scipy import constants
    >>> constants.value('elementary charge')
    1.602176634e-19

    """
    ...

def unit(key: str) -> str:
    """
    Unit in physical_constants indexed by key

    Parameters
    ----------
    key : Python string
        Key in dictionary `physical_constants`

    Returns
    -------
    unit : Python string
        Unit in `physical_constants` corresponding to `key`

    Examples
    --------
    >>> from scipy import constants
    >>> constants.unit('proton mass')
    'kg'

    """
    ...

def precision(key: str) -> float:
    """
    Relative precision in physical_constants indexed by key

    Parameters
    ----------
    key : Python string
        Key in dictionary `physical_constants`

    Returns
    -------
    prec : float
        Relative precision in `physical_constants` corresponding to `key`

    Examples
    --------
    >>> from scipy import constants
    >>> constants.precision('proton mass')
    5.1e-37

    """
    ...

def find(sub: str | None = ..., disp: bool = ...) -> Any:
    """
    Return list of physical_constant keys containing a given string.

    Parameters
    ----------
    sub : str
        Sub-string to search keys for. By default, return all keys.
    disp : bool
        If True, print the keys that are found and return None.
        Otherwise, return the list of keys without printing anything.

    Returns
    -------
    keys : list or None
        If `disp` is False, the list of keys is returned.
        Otherwise, None is returned.

    Examples
    --------
    >>> from scipy.constants import find, physical_constants

    Which keys in the ``physical_constants`` dictionary contain 'boltzmann'?

    >>> find('boltzmann')
    ['Boltzmann constant',
     'Boltzmann constant in Hz/K',
     'Boltzmann constant in eV/K',
     'Boltzmann constant in inverse meter per kelvin',
     'Stefan-Boltzmann constant']

    Get the constant called 'Boltzmann constant in Hz/K':

    >>> physical_constants['Boltzmann constant in Hz/K']
    (20836619120.0, 'Hz K^-1', 0.0)

    Find constants with 'radius' in the key:

    >>> find('radius')
    ['Bohr radius',
     'classical electron radius',
     'deuteron rms charge radius',
     'proton rms charge radius']
    >>> physical_constants['classical electron radius']
    (2.8179403262e-15, 'm', 1.3e-24)

    """
    ...

c = ...
mu0 = ...
epsilon0 = ...
exact_values = ...
_tested_keys = ...
