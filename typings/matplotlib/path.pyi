import numpy as np
from typing import Sequence
from ._typing import *
from .transforms import Affine2D, Bbox, Transform

from functools import lru_cache

class Path:
    code_type = np.uint8
    STOP = code_type(0)
    MOVETO = code_type(1)
    LINETO = code_type(2)
    CURVE3 = code_type(3)
    CURVE4 = code_type(4)
    CLOSEPOLY = code_type(79)
    NUM_VERTICES_FOR_CODE = ...
    def __init__(
        self,
        vertices: ArrayLike,
        codes: ArrayLike | None = ...,
        _interpolation_steps: int = ...,
        closed: bool = ...,
        readonly: bool = ...,
    ) -> None: ...
    @property
    def vertices(self) -> np.ndarray: ...
    @vertices.setter
    def vertices(self, vertices: ArrayLike): ...
    @property
    def codes(self) -> np.ndarray: ...
    @codes.setter
    def codes(self, codes: ArrayLike): ...
    @property
    def simplify_threshold(self): ...
    @simplify_threshold.setter
    def simplify_threshold(self, threshold): ...
    @property
    def should_simplify(self) -> bool: ...
    @should_simplify.setter
    def should_simplify(self, should_simplify: bool): ...
    @property
    def readonly(self) -> bool: ...
    def copy(self) -> Path: ...
    def __deepcopy__(self, memo=...) -> Path: ...
    deepcopy = ...
    @classmethod
    def make_compound_path_from_polys(cls, XY): ...
    @classmethod
    def make_compound_path(cls, *args): ...
    def __repr__(self): ...
    def __len__(self): ...
    def iter_segments(
        self,
        transform: None = ...,
        remove_nans: bool = ...,
        clip: None | float | float = ...,
        snap: None | bool = ...,
        stroke_width: float = ...,
        simplify: None | bool = ...,
        curves: bool = ...,
        sketch: None | Sequence = ...,
    ): ...
    def iter_bezier(self, **kwargs): ...
    def cleaned(
        self,
        transform: Transform = ...,
        remove_nans=...,
        clip=...,
        *,
        simplify=...,
        curves=...,
        stroke_width: float = ...,
        snap=...,
        sketch=...,
    ): ...
    def transformed(self, transform) -> Path: ...
    def contains_point(
        self,
        point: Sequence[float],
        transform: Transform = ...,
        radius: float = ...,
    ) -> bool: ...
    def contains_points(
        self, points: ArrayLike, transform: Transform = ..., radius: float = ...
    ) -> list[bool]: ...
    def contains_path(self, path: Path, transform: Transform = ...) -> bool: ...
    def get_extents(self, transform: Transform = ..., **kwargs) -> Bbox: ...
    def intersects_path(self, other: Path, filled: bool = ...) -> bool: ...
    def intersects_bbox(self, bbox: Bbox, filled: bool = ...): ...
    def interpolated(self, steps): ...
    def to_polygons(
        self,
        transform: Transform = ...,
        width: float = ...,
        height: float = ...,
        closed_only: bool = ...,
    ): ...
    @classmethod
    def unit_rectangle(cls) -> Path: ...
    @classmethod
    def unit_regular_polygon(cls, numVertices: int) -> Path: ...
    @classmethod
    def unit_regular_star(cls, numVertices: int, innerCircle=...) -> Path: ...
    @classmethod
    def unit_regular_asterisk(cls, numVertices: int) -> Path: ...
    @classmethod
    def unit_circle(cls) -> Path: ...
    @classmethod
    def circle(
        cls,
        center: Sequence[float] = ...,
        radius: float = ...,
        readonly: bool = ...,
    ) -> Path: ...
    @classmethod
    def unit_circle_righthalf(cls) -> Path: ...
    @classmethod
    def arc(
        cls, theta1: float, theta2: float, n: int = ..., is_wedge: bool = ...
    ) -> Path: ...
    @classmethod
    def wedge(cls, theta1: float, theta2: float, n: int = ...) -> Path: ...
    @staticmethod
    @lru_cache(8)
    def hatch(hatchpattern, density: float = ...): ...
    def clip_to_bbox(self, bbox: Bbox, inside: bool = ...): ...

def get_path_collection_extents(
    master_transform: Transform,
    paths: Sequence[Path],
    transforms: list,
    offsets,
    offset_transform: Affine2D,
) -> Bbox: ...
