from typing import Any, Callable, Literal, Sequence
from ._typing import *
from .transforms import Transform
from .axis import Axis

class ScaleBase:
    def __init__(self, axis: Axis) -> None: ...
    def get_transform(self) -> Transform: ...
    def set_default_locators_and_formatters(self, axis: Axis): ...
    def limit_range_for_scale(self, vmin: float, vmax: float, minpos: float): ...

class LinearScale(ScaleBase):
    name = ...
    def __init__(self, axis: Axis | None) -> None: ...
    def set_default_locators_and_formatters(self, axis: Axis): ...
    def get_transform(self) -> Transform: ...

class FuncTransform(Transform):
    input_dims = ...
    def __init__(self, forward: Callable, inverse: Callable) -> None: ...
    def transform_non_affine(self, values: ArrayLike) -> list: ...
    def inverted(self): ...

class FuncScale(ScaleBase):
    name = ...
    def __init__(self, axis: Axis, functions: Sequence[Callable]) -> None: ...
    def get_transform(self) -> FuncTransform: ...
    def set_default_locators_and_formatters(self, axis): ...

class LogTransform(Transform):
    input_dims = ...
    def __init__(self, base, nonpositive: Literal["clip", "mask"] = "clip") -> None: ...
    def __str__(self) -> str: ...
    def transform_non_affine(self, a) -> list: ...
    def inverted(self): ...

class InvertedLogTransform(Transform):
    input_dims = ...
    def __init__(self, base) -> None: ...
    def __str__(self) -> str: ...
    def transform_non_affine(self, a) -> list: ...
    def inverted(self): ...

class LogScale(ScaleBase):
    name = ...
    def __init__(
        self,
        axis: Axis,
        *,
        base: float = 10,
        subs=Sequence[int],
        nonpositive: Literal["clip", "mask"] = "clip",
    ) -> None: ...
    base = ...
    def set_default_locators_and_formatters(self, axis: Axis): ...
    def get_transform(self) -> LogTransform: ...
    def limit_range_for_scale(self, vmin: float, vmax: float, minpos: float): ...

class FuncScaleLog(LogScale):
    name = ...
    def __init__(
        self, axis: Axis, functions: Sequence[Callable], base: float = 10
    ) -> None: ...
    @property
    def base(self): ...
    def get_transform(self) -> Transform: ...

class SymmetricalLogTransform(Transform):
    input_dims = ...
    def __init__(self, base, linthresh, linscale) -> None: ...
    def transform_non_affine(self, a) -> list: ...
    def inverted(self): ...

class InvertedSymmetricalLogTransform(Transform):
    input_dims = ...
    def __init__(self, base, linthresh, linscale) -> None: ...
    def transform_non_affine(self, a) -> list: ...
    def inverted(self): ...

class SymmetricalLogScale(ScaleBase):
    name = ...
    def __init__(
        self,
        axis: Axis | None,
        *,
        base: float = 10,
        linthresh: float = 2,
        subs: Sequence[int] = ...,
        linscale: float = ...,
    ) -> None: ...

    base = ...
    linthresh = ...
    linscale = ...
    def set_default_locators_and_formatters(self, axis: Axis): ...
    def get_transform(self) -> SymmetricalLogTransform: ...

class AsinhTransform(Transform):
    input_dims = ...
    def __init__(self, linear_width) -> None: ...
    def transform_non_affine(self, a) -> list: ...
    def inverted(self): ...

class InvertedAsinhTransform(Transform):
    input_dims = ...
    def __init__(self, linear_width) -> None: ...
    def transform_non_affine(self, a) -> list: ...
    def inverted(self): ...

class AsinhScale(ScaleBase):
    name = ...
    auto_tick_multipliers = ...
    def __init__(
        self,
        axis: Axis,
        *,
        linear_width: float = 1,
        base: float = 10,
        subs: Sequence[int] = ...,
        **kwargs,
    ) -> None: ...
    linear_width = ...
    def get_transform(self): ...
    def set_default_locators_and_formatters(self, axis): ...

class LogitTransform(Transform):
    input_dims = ...
    def __init__(self, nonpositive: Literal["mask", "clip"] = ...) -> None: ...
    def transform_non_affine(self, a): ...
    def inverted(self): ...
    def __str__(self) -> str: ...

class LogisticTransform(Transform):
    input_dims = ...
    def __init__(self, nonpositive: Literal["mask", "clip"] = ...) -> None: ...
    def transform_non_affine(self, a): ...
    def inverted(self): ...
    def __str__(self) -> str: ...

class LogitScale(ScaleBase):
    name = ...
    def __init__(
        self,
        axis: Axis,
        nonpositive: Literal["mask", "clip"] = ...,
        *,
        one_half: str = ...,
        use_overline=...,
    ) -> None: ...
    def get_transform(self) -> LogitTransform: ...
    def set_default_locators_and_formatters(self, axis: Axis): ...
    def limit_range_for_scale(self, vmin: float, vmax: float, minpos: float): ...

def get_scale_names(): ...
def scale_factory(
    scale: Literal[
        "asinh", "function", "functionlog", "linear", "log", "logit", "symlog"
    ],
    axis: Axis,
    **kwargs,
): ...
def register_scale(scale_class): ...
