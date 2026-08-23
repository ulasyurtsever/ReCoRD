"""Importance-weight estimation over image embeddings.

Two estimators of the target/source density ratio evaluated at source
(calibration) points: a logistic domain classifier and a k-NN ratio. Both
operate on fixed image embeddings (e.g. DINOv2 or CLIP) and return clipped
weights; clipping bounds the variance of the weighted risk estimate at the
cost of bias, and the clip ceiling doubles as a conservative test weight for
weighted CRC.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def clip_weights(w: np.ndarray, clip: tuple[float, float]) -> np.ndarray:
    """Clip weights to ``[clip[0], clip[1]]``."""
    lo, hi = clip
    if not 0 < lo <= hi:
        raise ValueError(f"invalid clip range {clip}")
    return np.clip(w, lo, hi)


def logistic_density_ratio(
    source_emb: np.ndarray,
    target_emb: np.ndarray,
    clip: tuple[float, float] = (0.05, 20.0),
    c_reg: float = 1.0,
    random_state: int = 0,
    eval_emb: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate w(x) = p_target(x) / p_source(x) at the source points.

    A logistic classifier is trained to distinguish target (label 1) from
    source (label 0) embeddings; the odds ratio, corrected for class
    imbalance, estimates the density ratio.

    Returns
    -------
    np.ndarray
        Clipped weights of shape ``(len(source_emb),)``.
    """
    x = np.vstack([source_emb, target_emb])
    y = np.concatenate([np.zeros(len(source_emb)), np.ones(len(target_emb))])

    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(C=c_reg, max_iter=2000, random_state=random_state)
    clf.fit(scaler.transform(x), y)

    # ``eval_emb`` scores points the classifier was not fitted on, which is
    # what the disjoint-source-pool protocol needs.
    points = source_emb if eval_emb is None else eval_emb
    p = clf.predict_proba(scaler.transform(points))[:, 1]
    eps = 1e-12
    odds = p / np.clip(1.0 - p, eps, None)
    prior_correction = len(source_emb) / len(target_emb)
    return clip_weights(odds * prior_correction, clip)


def logistic_cv_density_ratio(
    source_emb: np.ndarray,
    target_emb: np.ndarray,
    clip: tuple[float, float] = (0.05, 20.0),
    c_reg: float = 1.0,
    random_state: int = 0,
    n_folds: int = 5,
) -> np.ndarray:
    """Cross-fitted version of :func:`logistic_density_ratio`.

    The in-sample estimator scores the very source points the classifier was
    trained on. In high dimension with few samples the classifier can separate
    the two domains almost perfectly on its training set, which drives the
    predicted target probability at every source point toward zero and the
    ratio toward the lower clip -- the same collapse that a genuine loss of
    support overlap produces. Out-of-fold probabilities remove that
    confound, so the two explanations can be told apart.

    Returns
    -------
    np.ndarray
        Clipped out-of-fold weights of shape ``(len(source_emb),)``.
    """
    from sklearn.model_selection import StratifiedKFold

    n_s, n_t = len(source_emb), len(target_emb)
    x = np.vstack([source_emb, target_emb])
    y = np.concatenate([np.zeros(n_s), np.ones(n_t)])
    folds = min(n_folds, int(min(n_s, n_t)))
    if folds < 2:
        raise ValueError(f"need at least 2 samples per domain, got {n_s}/{n_t}")

    p = np.full(n_s, np.nan)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True,
                               random_state=random_state)
    for train_idx, test_idx in splitter.split(x, y):
        scaler = StandardScaler().fit(x[train_idx])
        clf = LogisticRegression(C=c_reg, max_iter=2000,
                                 random_state=random_state)
        clf.fit(scaler.transform(x[train_idx]), y[train_idx])
        held_source = test_idx[test_idx < n_s]
        if held_source.size:
            p[held_source] = clf.predict_proba(
                scaler.transform(x[held_source]))[:, 1]

    if np.isnan(p).any():
        raise RuntimeError("cross-fitting left source points unscored")
    eps = 1e-12
    odds = p / np.clip(1.0 - p, eps, None)
    return clip_weights(odds * (n_s / n_t), clip)


def knn_density_ratio(
    source_emb: np.ndarray,
    target_emb: np.ndarray,
    k: int = 25,
    clip: tuple[float, float] = (0.05, 20.0),
) -> np.ndarray:
    """k-NN estimate of the density ratio at the source points.

    For each source point, the ratio of target to source neighbors among its
    ``k`` nearest neighbors in the pooled embedding set, corrected for pool
    sizes, estimates the density ratio.
    """
    n_s, n_t = len(source_emb), len(target_emb)
    if k >= n_s + n_t:
        raise ValueError(f"k={k} too large for pool of {n_s + n_t}")
    pool = np.vstack([source_emb, target_emb])
    is_target = np.concatenate([np.zeros(n_s, bool), np.ones(n_t, bool)])

    nn = NearestNeighbors(n_neighbors=k + 1).fit(pool)
    _, idx = nn.kneighbors(source_emb)
    idx = idx[:, 1:]  # drop self-match

    target_frac = is_target[idx].mean(axis=1)
    eps = 1e-12
    ratio = (target_frac / np.clip(1.0 - target_frac, eps, None)) * (n_s / n_t)
    return clip_weights(ratio, clip)
