# factor_metrics.py
# -*- coding: utf-8 -*-
import numpy as np

def evaluate_factor_soft_hard(
    probs_dict,
    main_attr_idx,
    other_attr_indices,
    source_label,
    target_label,
    main_delta_thresh=0.2,
    stable_delta_thresh=0.2,
    norm_stats=None,
):

    N = len(probs_dict)
    if N == 0:
        return {
            "SoftMain": 0.0,
            "SoftStable": 0.0,
            "HardMain": 0.0,
            "HardStable": 0.0,
            "N": 0,
        }

    if source_label == target_label:
        raise ValueError("source_label 和 target_label 必须不同（0->1 或 1->0）。")

    s = 1.0 if (source_label == 0 and target_label == 1) else -1.0

    soft_main_scores = []
    soft_stable_scores = []
    hard_main_flags = []
    hard_stable_flags = []

    for key, probs in probs_dict.items():
        arr = np.asarray(probs, dtype=np.float32)  # [T, num_attrs]
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue

        if norm_stats is not None:
            mn = norm_stats["min"]    # shape [num_attrs]
            mx = norm_stats["max"]    # shape [num_attrs]
            denom = (mx - mn) + 1e-6
            arr = (arr - mn[None, :]) / denom[None, :]
            arr = np.clip(arr, 0.0, 1.0)

        p_src = float(arr[0, main_attr_idx])
        p_tgt = float(arr[-1, main_attr_idx])

        delta_main = s * (p_tgt - p_src)

        edit_score = max(0.0, min(1.0, delta_main))
        soft_main_scores.append(edit_score)

        hard_main_flags.append(1 if delta_main >= main_delta_thresh else 0)

        stable_scores_this = []
        sample_stable_ok = True

        for j in other_attr_indices:
            q_src = float(arr[0, j])
            q_tgt = float(arr[-1, j])
            delta_j = abs(q_tgt - q_src)

            stable_score_j = 1.0 - delta_j
            stable_score_j = max(0.0, min(1.0, stable_score_j))
            stable_scores_this.append(stable_score_j)

            if delta_j > stable_delta_thresh:
                sample_stable_ok = False

        if stable_scores_this:
            soft_stable_scores.append(float(np.mean(stable_scores_this)))

        hard_stable_flags.append(1 if sample_stable_ok else 0)

    def _safe_mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    soft_main = _safe_mean(soft_main_scores)
    soft_stable = _safe_mean(soft_stable_scores)
    hard_main = _safe_mean(hard_main_flags)
    hard_stable = _safe_mean(hard_stable_flags)

    return {
        "SoftMain": soft_main,
        "SoftStable": soft_stable,
        "HardMain": hard_main,
        "HardStable": hard_stable,
        "N": N,
    }

