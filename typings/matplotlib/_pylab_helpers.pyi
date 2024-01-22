from .backend_bases import FigureManagerBase
from .figure import Figure
from typing import OrderedDict

from typing import Any, List

class Gcf:
    figs: OrderedDict = ...
    @classmethod
    def get_fig_manager(cls, num: int) -> None | FigureManagerBase: ...
    @classmethod
    def destroy(cls, num: int) -> None: ...
    @classmethod
    def destroy_fig(cls, fig: Figure) -> None: ...
    @classmethod
    def destroy_all(cls) -> None: ...
    @classmethod
    def has_fignum(cls, num: int) -> bool: ...
    @classmethod
    def get_all_fig_managers(cls) -> list[FigureManagerBase]: ...
    @classmethod
    def get_num_fig_managers(cls) -> int: ...
    @classmethod
    def get_active(cls) -> None | FigureManagerBase: ...
    @classmethod
    def set_active(cls, manager: FigureManagerBase): ...
    @classmethod
    def draw_all(cls, force: bool = False) -> None: ...
