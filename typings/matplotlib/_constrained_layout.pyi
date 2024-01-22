from matplotlib.backend_bases import RendererBase
from .figure import Figure
from .axes import Axes
from .transforms import Bbox

def do_constrained_layout(
    fig: Figure,
    h_pad: float,
    w_pad: float,
    hspace: float | None = None,
    wspace: float | None = None,
    rect: tuple = ...,
) -> dict: ...
def make_layoutgrids(fig, layoutgrids: dict, rect=...) -> dict: ...
def make_layoutgrids_gs(layoutgrids: dict, gs) -> dict: ...
def check_no_collapsed_axes(layoutgrids, fig: Figure) -> bool: ...
def compress_fixed_aspect(layoutgrids, fig: Figure): ...
def get_margin_from_padding(
    obj, *, w_pad: float = 0, h_pad: float = 0, hspace: float = 0, wspace: float = 0
) -> dict: ...
def make_layout_margins(
    layoutgrids: dict,
    fig: Figure,
    renderer: RendererBase,
    *,
    w_pad: float = 0,
    h_pad: float = 0,
    hspace: float = 0,
    wspace: float = 0,
) -> dict: ...
def make_margin_suptitles(
    layoutgrids: dict,
    fig: Figure,
    renderer: RendererBase,
    *,
    w_pad: float = 0,
    h_pad: float = 0,
) -> None: ...
def match_submerged_margins(layoutgrids: dict, fig: Figure) -> None: ...
def get_cb_parent_spans(cbax) -> tuple[range, range]: ...
def get_pos_and_bbox(ax: Axes, renderer: RendererBase) -> tuple[Bbox, Bbox]: ...
def reposition_axes(
    layoutgrids: dict,
    fig: Figure,
    renderer: RendererBase,
    *,
    w_pad: float = 0,
    h_pad: float = 0,
    hspace: float = 0,
    wspace: float = 0,
) -> None: ...
def reposition_colorbar(
    layoutgrids: dict,
    cbax: Axes,
    renderer: RendererBase,
    *,
    offset: float | None = None,
) -> dict: ...
def reset_margins(layoutgrids: dict, fig: Figure) -> None: ...
def colorbar_get_pad(layoutgrids: dict, cax: Axes) -> float: ...
