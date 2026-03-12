"""
Change-point detection evaluation metrics.

Implements:
- Hausdorff distance between two sets of change points
- True positives within a tolerance margin
- Precision
- Recall
- F1-score

The implementations follow the formulas provided by the user.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence
import math



def _to_sorted_list(points: Iterable[float]) -> List[float]:
    """
    Convert an iterable of change points to a sorted list of floats.
    """
    return sorted(float(x) for x in points)



def _min_distance(x: float, points: Sequence[float]) -> float:
    """
    Minimum absolute distance from x to a non-empty sequence of points.
    """
    return min(abs(x - p) for p in points)



def hausdorff_distance(
    true_cps: Iterable[float],
    pred_cps: Iterable[float],
) -> float:
    """
    Compute the Hausdorff distance between the true and predicted change-point sets.

    Definition:
        HAUSDORFF(T*, T_hat) = max{
            max_{t_hat in T_hat} min_{t* in T*} |t_hat - t*|,
            max_{t* in T*} min_{t_hat in T_hat} |t* - t_hat|
        }

    Parameters
    ----------
    true_cps : Iterable[float]
        Ground-truth change points.
    pred_cps : Iterable[float]
        Predicted change points.

    Returns
    -------
    float
        Hausdorff distance.

    Notes
    -----
    - If both sets are empty, returns 0.0.
    - If exactly one set is empty, returns math.inf.
    """
    true_list = _to_sorted_list(true_cps)
    pred_list = _to_sorted_list(pred_cps)

    if not true_list and not pred_list:
        return 0.0
    if not true_list or not pred_list:
        return math.inf

    term1 = max(_min_distance(t_hat, true_list) for t_hat in pred_list)
    term2 = max(_min_distance(t_star, pred_list) for t_star in true_list)
    return max(term1, term2)



def true_positives(
    true_cps: Iterable[float],
    pred_cps: Iterable[float],
    margin: float,
) -> List[float]:
    """
    Return the list of true change points that are detected within a margin.

    Definition:
        TP(T*, T_hat) = { t* in T* : exists t_hat in T_hat such that |t_hat - t*| < M }

    Parameters
    ----------
    true_cps : Iterable[float]
        Ground-truth change points.
    pred_cps : Iterable[float]
        Predicted change points.
    margin : float
        Error tolerance M. A true change point is counted as detected if a
        predicted change point lies at distance strictly less than margin.

    Returns
    -------
    List[float]
        The subset of true change points counted as true positives.
    """
    if margin < 0:
        raise ValueError("margin must be non-negative")

    true_list = _to_sorted_list(true_cps)
    pred_list = _to_sorted_list(pred_cps)

    if not true_list or not pred_list:
        return []

    matched_true = [
        t_star for t_star in true_list
        if any(abs(t_hat - t_star) < margin for t_hat in pred_list)
    ]
    return matched_true



def tp_count(
    true_cps: Iterable[float],
    pred_cps: Iterable[float],
    margin: float,
) -> int:
    """
    Number of true positives according to the TP definition above.
    """
    return len(true_positives(true_cps, pred_cps, margin))



def precision(
    true_cps: Iterable[float],
    pred_cps: Iterable[float],
    margin: float,
) -> float:
    """
    Compute the precision defined by:

        PREC(T*, T_hat) = |TP(T*, T_hat)| / K_hat

    where K_hat is the number of predicted change points.

    Returns 0.0 when there are no predicted change points.
    """
    pred_list = _to_sorted_list(pred_cps)
    k_hat = len(pred_list)
    if k_hat == 0:
        return 0.0
    return tp_count(true_cps, pred_list, margin) / k_hat



def recall(
    true_cps: Iterable[float],
    pred_cps: Iterable[float],
    margin: float,
) -> float:
    """
    Compute the recall defined by:

        REC(T*, T_hat) = |TP(T*, T_hat)| / K*

    where K* is the number of true change points.

    Returns 0.0 when there are no true change points.
    """
    true_list = _to_sorted_list(true_cps)
    k_star = len(true_list)
    if k_star == 0:
        return 0.0
    return tp_count(true_list, pred_cps, margin) / k_star



def f1_score(
    true_cps: Iterable[float],
    pred_cps: Iterable[float],
    margin: float,
) -> float:
    """
    Compute the F1-score:

        F1 = 2 * PREC * REC / (PREC + REC)

    Returns 0.0 if both precision and recall are zero.
    """
    prec = precision(true_cps, pred_cps, margin)
    rec = recall(true_cps, pred_cps, margin)

    if prec + rec == 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)



def evaluate_change_points(
    true_cps: Iterable[float],
    pred_cps: Iterable[float],
    margin: float,
) -> Dict[str, Any]:
    """
    Convenience function returning all metrics in one dictionary.
    """
    true_list = _to_sorted_list(true_cps)
    pred_list = _to_sorted_list(pred_cps)
    tp_list = true_positives(true_list, pred_list, margin)

    return {
        "true_change_points": true_list,
        "predicted_change_points": pred_list,
        "margin": float(margin),
        "hausdorff": hausdorff_distance(true_list, pred_list),
        "true_positives": tp_list,
        "tp_count": len(tp_list),
        "precision": precision(true_list, pred_list, margin),
        "recall": recall(true_list, pred_list, margin),
        "f1_score": f1_score(true_list, pred_list, margin),
    }


if __name__ == "__main__":
    true_cps = [50, 100, 150]
    pred_cps = [48, 103, 170]
    margin = 5

    results = evaluate_change_points(true_cps, pred_cps, margin)

    print("Change-point evaluation")
    print("-" * 30)
    for key, value in results.items():
        print(f"{key}: {value}")