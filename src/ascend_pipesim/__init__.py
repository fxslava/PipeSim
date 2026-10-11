"""ascend-pipesim: analytical flow and tiling simulator for Ascend AI Cores."""

__version__ = "0.2.0"

from .autotiler import AutotileOptions, AutotileResult, autotile, autotile_gemm
from .codegen import KernelBundle, synthesize, synthesize_kernel
from .dsl import (
    AlgorithmFlow,
    BufferPlan,
    EpilogueOp,
    ExecutionGraph,
    ExecutionMode,
    LoopNest,
    OpType,
    OutputPath,
    SimtWorkload,
    TensorSpec,
    TileConfig,
)
from .evaluator import EvaluationReport, EvalOptions, SimtEstimate, evaluate
from .frontier_search import (
    FrontierCandidate,
    FrontierResult,
    FrontierSearchOptions,
    find_optimal_frontiers,
)
from .parametric_evaluator import BoundedEvaluation, evaluate_parametric
from .uncertainty import EnvironmentScenario, Interval, UncertaintySpace
from .topology import (
    DType,
    NpuArchitecture,
    SimtHardware,
    get_architecture,
    list_architectures,
)

__all__ = [
    "__version__",
    # dsl
    "AlgorithmFlow",
    "BufferPlan",
    "EpilogueOp",
    "ExecutionGraph",
    "ExecutionMode",
    "LoopNest",
    "OpType",
    "OutputPath",
    "TileConfig",
    "SimtWorkload",
    "TensorSpec",
    # topology
    "DType",
    "NpuArchitecture",
    "SimtHardware",
    "get_architecture",
    "list_architectures",
    # evaluator
    "EvaluationReport",
    "EvalOptions",
    "SimtEstimate",
    "evaluate",
    "Interval",
    "EnvironmentScenario",
    "UncertaintySpace",
    "BoundedEvaluation",
    "evaluate_parametric",
    "FrontierCandidate",
    "FrontierResult",
    "FrontierSearchOptions",
    "find_optimal_frontiers",
    # autotiler
    "AutotileOptions",
    "AutotileResult",
    "autotile",
    "autotile_gemm",
    # codegen
    "KernelBundle",
    "synthesize",
    "synthesize_kernel",
]
