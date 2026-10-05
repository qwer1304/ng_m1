"""
utils.py -- small shared utilities (output formatting for notebook terminals).

=======================================================================
THE PROBLEM
=======================================================================
In Kaggle the terminal wraps log lines at a place that depends on the
browser's zoom, so long lines break unpredictably. "Virtual line breaks"
work around this: the text is written with a marker character (VB, a
backtick) where a break is wanted, and apply_virtual_breaks() turns each
marker into enough trailing spaces to make the terminal wrap there.

=======================================================================
CONTENTS
=======================================================================
VB                      the marker character used by this code base.
apply_virtual_breaks    text with markers -> text with padding (for stdout only).
strip_markers           text with markers -> plain text (for log files).
make_bar                tqdm progress bar with the Kaggle-safe settings.
set_bar_desc            set a bar's description through apply_virtual_breaks.

Usage
    desc = f"epoch [{ep}/{E}]{VB}loss={l:.4f} kl={k:.4f}"
    bar = make_bar(loader, ncols=80, bar=50)
    set_bar_desc(bar, desc, term_zoom=None)
    print(apply_virtual_breaks(line_with_markers, term_zoom, marker=VB))

Only text sent to the terminal gets the padding. Files (events.log,
log.jsonl) must get strip_markers(text) so they stay clean.
"""

from __future__ import annotations

import sys
from typing import Optional

VB = "`"  # the virtual-break marker of this code base

# terminal width (characters) by browser zoom factor (percent); verified values
ZOOM_MAP = {
    80: 191,
    90: 171,
    100: 140,
    110: 123,
}


def apply_virtual_breaks(text, zoom_factor=None, marker="|", overflow=200):
    """
    Replace every `marker` in `text` by padding that forces a line wrap.

    Each part between markers is followed by w spaces, where w is the terminal
    width for the given zoom (ZOOM_MAP), or `overflow` if zoom_factor is None
    (then the terminal is relied on to wrap and strip leading spaces). The
    padding forces the terminal to trip over the right margin and to strip the
    leading spaces of the next virtual line. We do not know how much space a
    part takes on its last line, so the padding is always the maximum.
    """
    assert zoom_factor is None or zoom_factor in ZOOM_MAP, \
        f"zoom factor {zoom_factor} not in database"
    w = overflow if zoom_factor is None else ZOOM_MAP[zoom_factor]
    padding = " " * w
    res = ""
    for part in text.split(marker):
        res += part + padding
    return res


def strip_markers(text: str, marker: str = VB, sep: str = " ") -> str:
    """Plain version of a marked text: every marker becomes `sep`."""
    return text.replace(marker, sep)


def make_bar(iterable, ncols: int = 80, bar: int = 50, total: Optional[int] = None,
             mininterval: float = 10.0, maxinterval: float = 20.0):
    """
    tqdm bar with fixed width and a slow refresh (the settings that behave in
    Kaggle). Progress is counted in minibatches.

    ncols        total width available (no autosizing).
    bar          length of the bar itself.
    mininterval  the display is refreshed at most every this many seconds.
    """
    from tqdm import tqdm

    bar_format = "{l_bar}{bar:" + str(bar) + "}{r_bar}"
    if total is None:
        total = len(iterable)
    return tqdm(
        iterable,
        total=total,
        ncols=ncols,
        dynamic_ncols=False,
        bar_format=bar_format,
        file=sys.stdout,
        mininterval=mininterval,
        maxinterval=maxinterval,
        ascii=True,
    )


def set_bar_desc(bar, text: str, term_zoom: Optional[int] = None, marker: str = VB) -> None:
    """
    Set the bar description, with the markers turned into virtual breaks.
    refresh=False: the new text shows at the next refresh allowed by
    mininterval; refresh=True would redraw (and, in a notebook, print a new
    line) on every call, i.e. on every minibatch.
    """
    bar.set_description(apply_virtual_breaks(text, term_zoom, marker=marker), refresh=False)
