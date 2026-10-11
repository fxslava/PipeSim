"""Pure DSL frontier API and JSON command-line entry point.

Usage: python -m ascend_pipesim.cli_or_api flow.json --arch ascend950pr
       --uncertainty uncertainty.json --top-k 5 > frontier.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .dsl import AlgorithmFlow
from .frontier_search import (
    FrontierSearchOptions,
    find_optimal_frontiers,
)
from .uncertainty import UncertaintySpace

__all__ = ["find_optimal_frontiers", "main"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Explore a parametric PipeSim frontier"
    )
    parser.add_argument("flow_file", type=Path, help="AlgorithmFlow JSON template")
    parser.add_argument("--arch", default="ascend950pr")
    parser.add_argument("--uncertainty", type=Path, help="JSON uncertainty box")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-evaluations", type=int, default=240)
    parser.add_argument("--max-tile", type=int, default=256)
    parser.add_argument("--forbid-tails", action="store_true")
    parser.add_argument(
        "--dominance-policy",
        choices=("scenario", "certified_interval"),
        default="scenario",
    )
    parser.add_argument("--output", type=Path, help="Write frontier JSON to this path")
    args = parser.parse_args(argv)
    flow = AlgorithmFlow.from_dict(
        json.loads(args.flow_file.read_text(encoding="utf-8"))
    )
    uncertainty = (
        UncertaintySpace.from_dict(
            json.loads(args.uncertainty.read_text(encoding="utf-8"))
        )
        if args.uncertainty
        else UncertaintySpace()
    )
    options = FrontierSearchOptions(
        top_k=args.top_k,
        max_evaluations=args.max_evaluations,
        max_tile=args.max_tile,
        forbid_tails=args.forbid_tails,
        dominance_policy=args.dominance_policy,
    )
    result = find_optimal_frontiers(flow, args.arch, uncertainty, options)
    payload = json.dumps(result.to_dict(), indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if result.candidates else 1


if __name__ == "__main__":
    raise SystemExit(main())
