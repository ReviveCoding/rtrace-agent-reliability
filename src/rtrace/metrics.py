from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def safe_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute PR-AUC without emitting undefined-class warnings on tiny fixtures.

    A split with no positive critical actions cannot support a meaningful PR-AUC.
    The local benchmark records `0.0` as a conservative non-winning value rather
    than silently treating the all-negative ranking as a perfect result.
    """
    label_array = np.asarray(labels, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    if label_array.size == 0 or int(label_array.sum()) == 0:
        return 0.0
    return float(average_precision_score(label_array, score_array))
