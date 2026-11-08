import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud Model Benchmark", layout="wide")
st.title("Fraud Detection: Validation to Test")
choices = [str(p) for p in Path(".").glob("*results.json")]
if not choices:
    st.error("Run python evaluate.py first")
    st.stop()
file = st.selectbox("Evaluation report", choices)
r = json.loads(Path(file).read_text())
st.info(r["source"])
st.caption(
    "Frozen test results. Model and threshold were selected using validation data only."
)
a, b, c = st.columns(3)
a.metric("Test average precision", f"{r['test']['average_precision']:.3f}")
b.metric("Test precision", f"{r['test']['precision']:.3f}")
c.metric("Test recall", f"{r['test']['recall']:.3f}")
st.write(
    "Selected model:",
    r["selected_model"],
    " | Threshold:",
    round(r["selected_threshold"], 5),
)
st.write("Split sizes (train / validation / test):", r["split_rows"])
st.subheader("Validation-only model comparison")
st.dataframe(pd.DataFrame(r["validation_comparison"]).T)
st.subheader("Untouched test confusion matrix")
st.dataframe(
    pd.DataFrame(
        r["test"]["confusion_matrix"],
        index=["Actual normal", "Actual fraud"],
        columns=["Predicted normal", "Predicted fraud"],
    )
)
st.subheader("Test precision-recall curve")
st.line_chart(
    pd.DataFrame(r["pr_curve"]).sort_values("recall"), x="recall", y="precision"
)
st.subheader("95% bootstrap intervals")
st.json(r["test_95_percent_bootstrap_ci"])
st.caption(r["ci_limit"])
st.subheader("Test permutation importance (AP decrease)")
st.bar_chart(
    pd.Series(r["permutation_importance_ap"]).sort_values(ascending=False).head(12)
)
st.json(r["costs"])
st.write("Test cost:", r["test_cost"])
st.write(r["limitations"])
st.download_button(
    "Download full report",
    json.dumps(r, indent=2),
    "fraud_report.json",
    "application/json",
)
