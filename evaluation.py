import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    precision_recall_curve,
)


def metrics(y, p, threshold):
    pred = np.asarray(p) >= threshold
    return dict(
        average_precision=float(average_precision_score(y, p)),
        roc_auc=float(roc_auc_score(y, p)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        brier=float(brier_score_loss(y, p)),
        confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    )


def choose_threshold(y, p, miss_cost=25, false_alarm_cost=1):
    candidates = np.unique(np.r_[0, np.quantile(p, np.linspace(0, 1, 501)), 1.000001])
    costs = []
    for t in candidates:
        pred = p >= t
        costs.append(
            miss_cost * np.sum((y == 1) & (~pred))
            + false_alarm_cost * np.sum((y == 0) & pred)
        )
    i = int(np.argmin(costs))
    return float(candidates[i]), float(costs[i])


def intervals(y, p, threshold, repeats=200):
    rng = np.random.default_rng(42)
    y = np.asarray(y)
    p = np.asarray(p)
    values = {k: [] for k in ["average_precision", "precision", "recall"]}
    for _ in range(repeats):
        ids = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ids])) < 2:
            continue
        m = metrics(y[ids], p[ids], threshold)
        for k in values:
            values[k].append(m[k])
    return {
        k: [float(x) for x in np.quantile(v, [0.025, 0.975])] for k, v in values.items()
    }


def curves(y, p):
    precision, recall, threshold = precision_recall_curve(y, p)
    return {"precision": precision.tolist(), "recall": recall.tolist()}
