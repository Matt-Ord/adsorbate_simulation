from matplotlib.widgets import Cursor
from typing import Any
from .backend_bases import Event, MouseEvent, ToolContainerBase
from .figure import Figure
from .axes import Axes
from .backend_managers import ToolManager

import enum

class Cursors(enum.IntEnum):
    POINTER: Cursors
    HAND: Cursors
    SELECT_REGION: Cursors
    MOVE: Cursors
    WAIT: Cursors
    RESIZE_HORIZONTAL: Cursors
    RESIZE_VERTICAL: Cursors

cursors = Cursors

class ToolBase:
    default_keymap: list[str] | None = None
    description: str | None = None
    image: str | None = None
    def __init__(self, toolmanager: ToolManager, name: str) -> None: ...

    name: property = ...
    toolmanager: property = ...
    canvas: property = ...
    @property
    def figure(self) -> Figure: ...
    @figure.setter
    def figure(self, figure: Figure) -> None: ...

    set_figure = figure.fset
    def trigger(self, sender: object, event: Event, data: object = ...) -> None: ...
    def destroy(self) -> None: ...

class ToolToggleBase(ToolBase):
    radio_group: str | None = ...
    cursor: Cursor | None = ...
    default_toggled: bool = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def trigger(self, sender, event, data=...) -> None: ...
    def enable(self, event=...) -> None: ...
    def disable(self, event=...) -> None: ...
    @property
    def toggled(self) -> bool: ...
    def set_figure(self, figure: Figure) -> None: ...

class SetCursorBase(ToolBase):
    def __init__(self, *args, **kwargs) -> None: ...
    def set_figure(self, figure: Figure) -> None: ...
    def set_cursor(self, cursor: Cursor) -> None: ...

ToolSetCursor = SetCursorBase

class ToolCursorPosition(ToolBase):
    def __init__(self, *args, **kwargs) -> None: ...
    def set_figure(self, figure: Figure) -> None: ...
    def send_message(self, event: MouseEvent) -> None: ...

class RubberbandBase(ToolBase):
    def trigger(self, sender, event: Event, data) -> None: ...
    def draw_rubberband(self, *data) -> None: ...
    def remove_rubberband(self) -> None: ...

class ToolQuit(ToolBase):
    description: str = ...
    default_keymap: list[str] = ...
    def trigger(self, sender: object, event: Event, data: object = ...) -> None: ...

class ToolQuitAll(ToolBase):
    description: str = ...
    default_keymap: list[str] = ...
    def trigger(self, sender: object, event: Event, data: object = ...) -> None: ...

class ToolGrid(ToolBase):
    description: str = ...
    default_keymap: list[str] = ...
    def trigger(self, sender: object, event: Event, data: object = ...) -> None: ...

class ToolMinorGrid(ToolBase):
    description: str = ...
    default_keymap: list[str] = ...
    def trigger(self, sender: object, event: Event, data: object = ...) -> None: ...

class ToolFullScreen(ToolBase):
    description: str = ...
    default_keymap: list[str] = ...
    def trigger(self, sender: object, event: Event, data: object = ...) -> None: ...

class AxisScaleBase(ToolToggleBase):
    def trigger(self, sender, event: Event, data=...) -> None: ...
    def enable(self, event: Event) -> None: ...
    def disable(self, event: Event) -> None: ...

class ToolYScale(AxisScaleBase):
    description: str = ...
    default_keymap: list[str] = ...
    def set_scale(self, ax, scale) -> None: ...

class ToolXScale(AxisScaleBase):
    description: str = ...
    default_keymap: list[str] = ...
    def set_scale(self, ax: Axes, scale) -> None: ...

class ToolViewsPositions(ToolBase):
    def __init__(self, *args, **kwargs) -> None: ...
    def add_figure(self, figure: Figure) -> None: ...
    def clear(self, figure: Figure) -> None: ...
    def update_view(self) -> None: ...
    def push_current(self, figure: Figure = ...) -> None: ...
    def update_home_views(self, figur: Figure = ...) -> None: ...
    def home(self) -> None: ...
    def back(self) -> None: ...
    def forward(self) -> None: ...

class ViewsPositionsBase(ToolBase):
    def trigger(self, sender: object, event: Event, data: object = ...) -> None: ...

class ToolHome(ViewsPositionsBase):
    description: str = ...
    image: str = ...
    default_keymap: list[str] = ...

class ToolBack(ViewsPositionsBase):
    description: str = ...
    image: str = ...
    default_keymap: list[str] = ...

class ToolForward(ViewsPositionsBase):
    description: str = ...
    image: str = ...
    default_keymap: list[str] = ...

class ConfigureSubplotsBase(ToolBase):
    description: str = ...
    image: str = ...

class SaveFigureBase(ToolBase):
    description: str = ...
    image: str = ...
    default_keymap: list[str] = ...

class ZoomPanBase(ToolToggleBase):
    def __init__(self, *args) -> None: ...
    def enable(self, event: Event) -> None: ...
    def disable(self, event: Event) -> None: ...
    def trigger(self, sender, event: Event, data=...) -> None: ...
    def scroll_zoom(self, event: Event) -> None: ...

class ToolZoom(ZoomPanBase):
    description: str = ...
    image: str = ...
    default_keymap: list[str] = ...
    cursor: Cursors = ...
    radio_group: str = ...
    def __init__(self, *args) -> None: ...

class ToolPan(ZoomPanBase):
    default_keymap: list[str] = ...
    description: str = ...
    image: str = ...
    cursor: Cursors = ...
    radio_group: str = ...
    def __init__(self, *args) -> None: ...

class ToolHelpBase(ToolBase):
    description: str = ...
    default_keymap: list[str] = ...
    image: str = ...
    @staticmethod
    def format_shortcut(key_sequence) -> None: ...

class ToolCopyToClipboardBase(ToolBase):
    description: str = ...
    default_keymap: list[str] = ...
    def trigger(self, *args, **kwargs) -> None: ...

default_tools: dict[str, ToolBase] = ...
default_toolbar_tools: list[list[str | list[str]]] = ...

def add_tools_to_manager(
    toolmanager: ToolManager, tools: dict[str, Any] = ...
) -> None: ...
def add_tools_to_container(container: ToolContainerBase, tools: list = ...) -> None: ...
