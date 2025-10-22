# pycpet/intro_citation.py
from __future__ import annotations
import os
import sys
import shutil

# --- Config knobs ------------------------------------------------------------
# CPET_BANNER=0    -> force disable
# CPET_BANNER=1    -> force enable
# Default: enabled (prints on import), except auto-disabled under pytest.

_SHOWN = False  # process-local guard


def _should_show_banner() -> bool:
    env = os.getenv("CPET_BANNER")
    if env is not None:
        return env.strip() == "1"

    # Default behavior: show, unless under pytest
    under_pytest = "PYTEST_CURRENT_TEST" in os.environ
    return not under_pytest


def _line(width: int) -> str:
    return "=" * max(10, width)  # minimum width so it always looks OK


def _print_banner() -> None:
    global _SHOWN
    if _SHOWN:
        return
    _SHOWN = True

    try:
        width = shutil.get_terminal_size().columns
    except Exception:
        width = 72

    top = "\n" + _line(min(72, width)) + "\n"
    bottom = _line(min(72, width)) + "\n"

    msg = (
        "CPET: Computing and Analyzing Electric Fields in Proteins\n"
        "Developed by: Pujan Ajmera, Santiago Vargas, Matthew Hennerfarth, and Anastassia Alexandrova\n"
        "Maintainer: pujanajmera1@g.ucla.edu\n"
        "GitHub: https://github.com/pujanajmera/pycpet\n"
        "License: MIT — please see repository for full license text\n"
        "For citation: 10.1021/acs.jctc.5c00138\n"
    )

    citation = (
        "Mandatory citation if you use this code: \n"
        "- 10.1021/acs.jctc.5c00138\n"
        "Preferred citations: \n"
        "If you are using the tensor decomposition method, please also cite: \n"
        "- 10.1021/jacs.5c11931\n"
        "If you are using the distribution of streamlines method, please also cite: \n"
        "- 10.1021/acscatal.0c02795\n"
        "If you are using PCA to decompose electric fields, please also cite: \n"
        "- 10.1021/jacs.4c03914\n"
    )

    try:
        sys.stdout.write(top + msg + bottom + citation)
        sys.stdout.flush()
    except Exception:
        # Never let the banner break imports
        pass


# --- Run on import (side effect) --------------------------------------------
if _should_show_banner():
    _print_banner()
