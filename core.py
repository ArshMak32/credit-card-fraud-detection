from threadpoolctl import threadpool_limits
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from evaluation import metrics, choose_threshold, intervals, curves


def load_data(path=None):
    if path:
        df = pd.read_csv(path)
        needed = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
        if not needed.issubset(df):
            raise ValueError("Expected ULB columns: Time, Amount, V1..V28, Class")
        df = (
            df[list(sorted(needed))]
            .drop_duplicates()
            .sort_values("Time", kind="stable")
            .reset_index(drop=True)
        )
        if df.isna().any().any() or not np.isfinite(df.to_numpy()).all():
            raise ValueError("Data must be finite and complete")
        if set(df.Class.unique()) != {0, 1}:
            raise ValueError("Class must contain 0 and 1")
        # Time is split metadata only. Anonymized PCA was performed upstream, outside our control.
        X = df.drop(columns=["Class", "Time"])
        y = df.Class.to_numpy(dtype=int)
        source = "ULB credit-card benchmark (real anonymized transactions)"
    else:
        a, y = make_classification(
            n_samples=9000,
            n_features=12,
            n_informative=7,
            n_redundant=2,
            weights=[0.98, 0.02],
            flip_y=0.002,
            class_sep=1.5,
            random_state=42,
        )
        X = pd.DataFrame(a, columns=[f"feature_{i}" for i in range(12)])
        source = "SYNTHETIC DEMO - not real credit-card data"
    if len(X) < 300:
        raise ValueError("At least 300 rows required")
    return X, y, source


def partitions(X, y):
    # Consecutive 60/20/20 blocks: preserves chronology for the real dataset.
    n = len(y)
    a = int(n * 0.6)
    b = int(n * 0.8)
    chunks = (slice(0, a), slice(a, b), slice(b, n))
    if any(len(np.unique(y[s])) != 2 for s in chunks):
        raise ValueError("Every time block must contain both classes")
    return [(X.iloc[s], y[s]) for s in chunks]


@threadpool_limits.wrap(limits=2)
def run(path=None, miss_cost=25, false_alarm_cost=1):
    if miss_cost <= 0 or false_alarm_cost <= 0:
        raise ValueError("Costs must be positive")
    X, y, source = load_data(path)
    (xt, yt), (xv, yv), (xs, ys) = partitions(X, y)
    models = {
        "prior_baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=2,
            random_state=42,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=15,
            l2_regularization=2,
            early_stopping=False,
            random_state=42,
        ),
    }
    validation = {}
    fitted = {}
    for name, m in models.items():
        m.fit(xt, yt)
        p = m.predict_proba(xv)[:, 1]
        threshold, cost = choose_threshold(yv, p, miss_cost, false_alarm_cost)
        validation[name] = {
            "threshold": threshold,
            "validation_cost": cost,
            **metrics(yv, p, threshold),
        }
        fitted[name] = m
    winner = min(validation, key=lambda k: validation[k]["validation_cost"])
    model = fitted[winner]
    threshold = validation[winner]["threshold"]
    p = model.predict_proba(xs)[:, 1]
    pred = p >= threshold
    perm = permutation_importance(
        model,
        xs,
        ys,
        scoring="average_precision",
        n_repeats=3,
        max_samples=min(3000, len(xs)),
        random_state=42,
        n_jobs=2,
    )
    result = {
        "source": source,
        "split": "ordered 60% train / 20% validation / 20% test; duplicates removed before real-data split",
        "rows": len(y),
        "split_rows": [len(yt), len(yv), len(ys)],
        "positive_counts": [int(sum(yt)), int(sum(yv)), int(sum(ys))],
        "costs": {
            "missed_fraud": miss_cost,
            "false_alarm": false_alarm_cost,
            "units": "hypothetical review-cost units, not dollars",
        },
        "selection": "lowest validation cost; frozen model and threshold evaluated once on test",
        "selected_model": winner,
        "selected_threshold": threshold,
        "validation_comparison": validation,
        "test": metrics(ys, p, threshold),
        "test_cost": float(
            miss_cost * np.sum((ys == 1) & (~pred))
            + false_alarm_cost * np.sum((ys == 0) & pred)
        ),
        "test_95_percent_bootstrap_ci": intervals(ys, p, threshold),
        "ci_limit": "IID bootstrap; does not model temporal dependence",
        "permutation_importance_ap": dict(
            zip(X.columns, perm.importances_mean.tolist())
        ),
        "pr_curve": curves(ys, p),
        "limitations": [
            "Only two days in ULB benchmark; no independent bank validation",
            "Upstream PCA preprocessing is fixed and cannot be audited for leakage here",
            "Costs are illustrative, scores uncalibrated, and feature meaning is anonymized",
        ],
    }
    predictions = xs.copy()
    predictions["actual"] = ys
    predictions["score"] = p
    predictions["flagged"] = pred
    return result, predictions
