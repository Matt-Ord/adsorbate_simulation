from typing import Sequence
from matplotlib.path import Path
from matplotlib.transforms import Transform
from matplotlib.backend_bases import MouseButton
from matplotlib.ticker import Formatter
from matplotlib.axes import Axes
import numpy as np

class GeoAxes(Axes):
    class ThetaFormatter(Formatter):
        def __init__(self, round_to=...) -> None: ...
        def __call__(self, x, pos=...): ...

    RESOLUTION = ...
    def clear(self) -> None: ...
    def get_xaxis_transform(self, which=...) -> Transform: ...
    def get_xaxis_text1_transform(self, pad) -> Transform: ...
    def get_xaxis_text2_transform(self, pad) -> Transform: ...
    def get_yaxis_transform(self, which=...) -> Transform: ...
    def get_yaxis_text1_transform(self, pad) -> Transform: ...
    def get_yaxis_text2_transform(self, pad) -> Transform: ...
    def set_yscale(self, *args, **kwargs) -> None: ...

    set_xscale = set_yscale
    def set_xlim(self, *args, **kwargs) -> None: ...
    set_ylim = set_xlim
    def format_coord(self, lon, lat) -> str: ...
    def set_longitude_grid(self, degrees: float) -> None: ...
    def set_latitude_grid(self, degrees: float) -> None: ...
    def set_longitude_grid_ends(self, degrees: float) -> None: ...
    def get_data_ratio(self) -> float: ...
    def can_zoom(self) -> bool: ...
    def can_pan(self) -> bool: ...
    def start_pan(self, x: float, y: float, button: MouseButton): ...
    def end_pan(self) -> bool: ...
    def drag_pan(self, button: MouseButton, key: str | None, x: float, y: float): ...

class _GeoTransform(Transform):
    input_dims = ...
    def __init__(self, resolution) -> None: ...
    def __str__(self) -> str: ...
    def transform_path_non_affine(self, path: Path) -> Path: ...

class AitoffAxes(GeoAxes):
    name: str = ...

    class AitoffTransform(_GeoTransform):
        def transform_non_affine(self, ll) -> np.ndarray: ...
        def inverted(self) -> _GeoTransform: ...

    class InvertedAitoffTransform(_GeoTransform):
        def transform_non_affine(self, xy: Sequence[float]): ...
        def inverted(self) -> _GeoTransform: ...

    def __init__(self, *args, **kwargs) -> None: ...

class HammerAxes(GeoAxes):
    name = ...

    class HammerTransform(_GeoTransform):
        def transform_non_affine(self, ll) -> np.ndarray: ...
        def inverted(self) -> _GeoTransform: ...

    class InvertedHammerTransform(_GeoTransform):
        def transform_non_affine(self, xy: Sequence[float]) -> np.ndarray: ...
        def inverted(self) -> _GeoTransform: ...

    def __init__(self, *args, **kwargs) -> None: ...

class MollweideAxes(GeoAxes):
    name = ...

    class MollweideTransform(_GeoTransform):
        def transform_non_affine(self, ll) -> np.ndarray: ...
        def inverted(self) -> _GeoTransform: ...

    class InvertedMollweideTransform(_GeoTransform):
        def transform_non_affine(self, xy: Sequence[float]) -> np.ndarray: ...
        def inverted(self) -> _GeoTransform: ...

    def __init__(self, *args, **kwargs) -> None: ...

class LambertAxes(GeoAxes):
    name = ...

    class LambertTransform(_GeoTransform):
        def __init__(self, center_longitude, center_latitude, resolution) -> None: ...
        def transform_non_affine(self, ll) -> np.ndarray: ...
        def inverted(self) -> _GeoTransform: ...

    class InvertedLambertTransform(_GeoTransform):
        def __init__(self, center_longitude, center_latitude, resolution) -> None: ...
        def transform_non_affine(self, xy: Sequence[float]) -> np.ndarray: ...
        def inverted(self) -> _GeoTransform: ...

    def __init__(
        self, *args, center_longitude=..., center_latitude=..., **kwargs
    ) -> None: ...
    def clear(self) -> None: ...
