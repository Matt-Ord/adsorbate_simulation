import numpy as np
from typing import Any, Callable, Generator, Iterator, overload
from matplotlib._typing import *
from matplotlib.artist import Artist

import collections
import collections.abc
import contextlib

class _StrongRef:
    def __init__(self, obj) -> None: ...
    def __call__(self): ...
    def __eq__(self, other) -> bool: ...
    def __hash__(self) -> int: ...

class CallbackRegistry:
    def __init__(
        self, exception_handler: Callable = ..., *, signals: list = ...
    ) -> None: ...
    def __getstate__(self) -> dict[str, Any]: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...
    def connect(self, signal, func: Callable) -> int: ...
    def disconnect(self, cid: int) -> None: ...
    def process(self, s: str, *args, **kwargs) -> None: ...
    @contextlib.contextmanager
    def blocked(self, *, signal: str = ...) -> Generator: ...

class silent_list(list):
    def __init__(self, type, seq=...) -> None: ...
    def __repr__(self) -> str: ...

def strip_math(s: str) -> str: ...
def is_writable_file_like(obj) -> bool: ...
def file_requires_unicode(x) -> bool: ...
def to_filehandle(
    fname: str | PathLike | FileLike,
    flag: str = "r",
    return_opened: bool = False,
    encoding: str | None = None,
) -> tuple[FileLike, bool]: ...
def open_file_cm(
    path_or_file: str | PathLike | FileLike, mode=..., encoding=...
) -> FileLike: ...
def is_scalar_or_string(val) -> bool: ...
@overload
def get_sample_data(
    fname, asfileobj: bool = ..., *, np_load: bool = ...
) -> ArrayLike | tuple | dict: ...
@overload
def get_sample_data(fname, asfileobj: bool, *, np_load=...) -> PathLike: ...
@overload
def get_sample_data(fname, asfileobj: bool = ..., *, np_load=...) -> FileLike: ...
def flatten(seq, scalarp=...) -> Generator: ...

class maxdict(dict):
    def __init__(self, maxsize) -> None: ...
    def __setitem__(self, k, v) -> None: ...

class Stack:
    def __init__(self, default=...) -> None: ...
    def __call__(self): ...
    def __len__(self) -> int: ...
    def __getitem__(self, ind): ...
    def forward(self): ...
    def back(self): ...
    def push(self, o): ...
    def home(self): ...
    def empty(self) -> bool: ...
    def clear(self) -> None: ...
    def bubble(self, o): ...
    def remove(self, o) -> None: ...

def report_memory(i=...):
    int: ...

def safe_masked_invalid(x, copy=...) -> np.ndarray: ...
def print_cycles(objects, outstream=..., show_progress: bool = ...) -> None: ...

class Grouper:
    def __init__(self, init=...) -> None: ...
    def __contains__(self, item) -> bool: ...
    def clean(self) -> None: ...
    def join(self, a, *args) -> bool: ...
    def joined(self, a, b) -> bool: ...
    def remove(self, a) -> None: ...
    def __iter__(self) -> Generator: ...
    def get_siblings(self, a) -> list: ...

class GrouperView:
    def __init__(self, grouper) -> None: ...

    class _GrouperMethodForwarder:
        def __init__(self, deprecated_kw=...) -> None: ...
        def __set_name__(self, owner, name): ...

    joined = ...
    get_siblings = ...
    clean = ...
    join = ...
    remove = ...

def simple_linear_interpolation(a, steps: int) -> np.ndarray: ...
def delete_masked_points(*args) -> tuple: ...
def boxplot_stats(
    X: ArrayLike,
    whis: float = 1.5,
    bootstrap: int = ...,
    labels: ArrayLike = ...,
    autorange: bool = False,
) -> list[dict]: ...

ls_mapper: dict[str, str] = ...
ls_mapper_r: dict[str, str] = ...

def contiguous_regions(mask) -> list[tuple]: ...
def is_math_text(s) -> bool: ...
def violin_stats(
    X: ArrayLike, method: Callable, points: int = ..., quantiles: ArrayLike = ...
) -> list[dict]: ...
def pts_to_prestep(x: list, *args) -> list: ...
def pts_to_poststep(x: list, *args) -> list: ...
def pts_to_midstep(x: list, *args) -> list: ...

STEP_LOOKUP_MAP: dict = ...

def index_of(
    y: float | ArrayLike,
) -> tuple[np.ndarray, np.ndarray]: ...
def safe_first_element(obj): ...
def sanitize_sequence(data) -> list: ...
def normalize_kwargs(kw: dict | None, alias_mapping: dict | Artist = ...) -> dict: ...

class _OrderedSet(collections.abc.MutableSet):
    def __init__(self) -> None: ...
    def __contains__(self, key) -> bool: ...
    def __iter__(self) -> Iterator: ...
    def __len__(self) -> int: ...
    def add(self, key) -> None: ...
    def discard(self, key) -> None: ...
