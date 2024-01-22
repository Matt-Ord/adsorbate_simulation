from typing import Type
from matplotlib._typing import *
from matplotlib.transforms import Transform
from matplotlib.text import Text
from matplotlib.backend_bases import (
    _Backend,
    FigureCanvasBase,
    FigureManagerBase,
    GraphicsContextBase,
)
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Affine2DBase

from enum import Enum

from traitlets import Int
from . import _backend_pdf_ps

backend_version: str = ...
debugPS: bool = ...

class PsBackendHelper:
    def __init__(self) -> None: ...

ps_backend_helper: PsBackendHelper = ...
papersize: dict[str, tuple[float, float]] = ...

def quote_ps_string(s: str) -> str: ...

class RendererPS(_backend_pdf_ps.RendererPDFPSBase):
    def __init__(self, width, height, pswriter, imagedpi=...) -> None: ...
    def set_color(
        self, r: float, g: float, b: float, store: float | bool = ...
    ) -> None: ...
    def set_linewidth(self, linewidth: float, store: bool = ...) -> None: ...
    def set_linejoin(self, linejoin: str, store: bool = ...) -> None: ...
    def set_linecap(self, linecap: str, store: bool = ...) -> None: ...
    def set_linedash(self, offset: int, seq: None, store: bool = ...) -> None: ...
    def set_font(self, fontname: str, fontsize: float, store: bool = ...) -> None: ...
    def create_hatch(self, hatch) -> str: ...
    def get_image_magnification(self) -> float: ...
    def draw_image(
        self,
        gc: GraphicsContextBase,
        x: Scalar,
        y: Scalar,
        im,
        transform: Affine2DBase = ...,
    ) -> None: ...
    def draw_path(self, gc, path, transform, rgbFace=...) -> None: ...
    def draw_markers(
        self,
        gc: GraphicsContextBase,
        marker_path,
        marker_trans: Transform,
        path,
        trans: Transform,
        rgbFace=...,
    ) -> None: ...
    def draw_path_collection(
        self,
        gc,
        master_transform,
        paths,
        all_transforms,
        offsets,
        offsetTrans,
        facecolors,
        edgecolors,
        linewidths,
        linestyles,
        antialiaseds,
        urls,
        offset_position,
    ) -> None: ...
    def draw_tex(self, gc, x, y, s, prop, angle, *, mtext=...) -> None: ...
    def draw_text(
        self,
        gc: GraphicsContextBase,
        x: float,
        y: float,
        s: str,
        prop: FontProperties,
        angle: float,
        ismath=...,
        mtext: Text = ...,
    ) -> None: ...
    def draw_mathtext(self, gc, x, y, s, prop, angle) -> None: ...
    def draw_gouraud_triangle(
        self, gc: GraphicsContextBase, points, colors, trans
    ) -> None: ...
    def draw_gouraud_triangles(self, gc, points, colors, trans) -> None: ...

class _Orientation(Enum):
    def swap_if_landscape(self, shape: tuple[float, int]) -> tuple[float, int]: ...

class FigureCanvasPS(FigureCanvasBase):
    fixed_dpi: Int = ...
    filetypes: dict[str, str] = ...
    def get_default_filetype(self) -> str: ...
    def print_ps(
        self,
        outfile,
        *args,
        metadata=None,
        papertype=None,
        orientation="portrait",
        **kwargs,
    ) -> None: ...
    def print_eps(
        self,
        outfile,
        *args,
        metadata=None,
        papertype=None,
        orientation="portrait",
        **kwargs,
    ) -> None: ...
    def draw(self) -> None: ...

def convert_psfrags(
    tmpfile,
    psfrags,
    font_preamble,
    custom_preamble,
    paper_width,
    paper_height,
    orientation,
) -> bool: ...
def gs_distill(tmpfile, eps=..., ptype=..., bbox=..., rotated=...) -> None: ...
def xpdf_distill(tmpfile, eps=..., ptype=..., bbox=..., rotated=...) -> None: ...
def get_bbox_header(lbrt, rotated=...) -> tuple[str, str]: ...
def pstoeps(tmpfile, bbox=..., rotated=...) -> None: ...

FigureManagerPS: Type[FigureManagerBase] = ...
psDefs: list[str] = ...

class _BackendPS(_Backend):
    FigureCanvas = FigureCanvasPS
