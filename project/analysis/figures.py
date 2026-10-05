"""Opt-in figure auto-save: nothing is written until a notebook calls set_figure_dir().

    set_figure_dir(FIGURES_DIR / cfg.name)   # -> data/figures/<dataset>/<plot name>.png

Off by default so an exploratory call cannot overwrite a thesis figure. `save_path=` always wins.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_DPI = 300          # print resolution
DEFAULT_FORMAT = "png"

_state: dict = {"dir": None, "dpi": DEFAULT_DPI, "format": DEFAULT_FORMAT}


def set_figure_dir(path=None, dpi: int = DEFAULT_DPI, fmt: str = DEFAULT_FORMAT) -> Path | None:
    """Turn auto-saving on and create the directory; set_figure_dir(None) turns it off.

    `fmt` is any format matplotlib writes; 'pdf'/'svg' give vector output.
    """
    _state["dpi"], _state["format"] = dpi, fmt
    if path is None:
        _state["dir"] = None
        return None
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    _state["dir"] = directory
    return directory


def figure_dir() -> Path | None:
    """The active auto-save directory, or None when auto-saving is off."""
    return _state["dir"]


def figure_path(name: str | None):
    """Destination for a figure called `name`, or None when auto-saving is off."""
    if not name or _state["dir"] is None:
        return None
    return _state["dir"] / f"{name}.{_state['format']}"


def save_figure(fig, path, extra_artists=None) -> None:
    """Write `fig` at the configured dpi.

    bbox_inches='tight' crops to the axes and would truncate legends anchored outside them,
    so they are passed explicitly.
    """
    fig.savefig(path, dpi=_state["dpi"], bbox_inches="tight",
                bbox_extra_artists=list(extra_artists) if extra_artists else None)
