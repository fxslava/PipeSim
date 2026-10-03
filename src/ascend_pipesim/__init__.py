"""ascend-pipesim: analytical flow and tiling simulator for Ascend AI Cores."""

__version__ = "0.2.0"

from .autotiler import AutotileOptions, AutotileResult, autotile, autotile_gemm
from .codegen import KernelBundle, synthesize, synthesize_kernel
from .dsl import (
    AlgorithmFlow,
    BufferPlan,
    EpilogueOp,
    LoopNest,
    OpType,
    TileConfig,
)
from .evaluator import EvaluationReport, EvalOptions, evaluate
from .topology import (
    DType,
    NpuArchitecture,
    get_architecture,
    list_architectures,
)

__all__ = [
    "__version__",
    # dsl
    "AlgorithmFlow", "BufferPlan", "EpilogueOp", "LoopNest", "OpType",
    "TileConfig",
    # topology
    "DType", "NpuArchitecture", "get_architecture", "list_architectures",
    # evaluator
    "EvaluationReport", "EvalOptions", "evaluate",
    # autotiler
    "AutotileOptions", "AutotileResult", "autotile", "autotile_gemm",
    # codegen
    "KernelBundle", "synthesize", "synthesize_kernel",
]
