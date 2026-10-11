"""Compatibility entry point for frontier search.

The command-line interface lives in :mod:`ascend_pipesim.cli`; prefer
``npu-pipesim frontier`` for new usage.
"""

from __future__ import annotations

import sys
import warnings
from typing import Sequence

from .cli import main as _cli_main
from .cli import run_frontier
from .frontier_search import find_optimal_frontiers

__all__ = ["find_optimal_frontiers", "main", "run_frontier"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the integrated frontier command through the deprecated entry point."""
    warnings.warn(
        "ascend_pipesim.cli_or_api is deprecated; use "
        "`npu-pipesim frontier` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and not arguments[0].startswith("-"):
        arguments[:1] = ["--flow-file", arguments[0]]
    return _cli_main(["frontier", *arguments])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
