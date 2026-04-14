"""AdamOpt - Automatic text frequency optimization for LLMs based on Adam's Law."""

__version__ = "0.1.0"

from adamopt.content_locker import ContentLocker, LockResult
from adamopt.frequency import FrequencyEstimator, FrequencyResult, FrequencySource
from adamopt.optimizer import OptimizeResult, TextOptimizer
from adamopt.tfd import TFDDistiller

__all__ = [
    "ContentLocker",
    "FrequencyEstimator",
    "FrequencyResult",
    "FrequencySource",
    "LockResult",
    "OptimizeResult",
    "TFDDistiller",
    "TextOptimizer",
    "__version__",
]
