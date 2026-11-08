import argparse, json
from pathlib import Path
from core import run

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data")
    p.add_argument("--miss-cost", type=float, default=25)
    p.add_argument("--false-alarm-cost", type=float, default=1)
    p.add_argument("--output", default="results.json")
    a = p.parse_args()
    report, pred = run(a.data, a.miss_cost, a.false_alarm_cost)
    Path(a.output).write_text(json.dumps(report, indent=2) + "\n")
    pred.to_csv(Path(a.output).with_suffix(".predictions.csv"), index=False)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ["pr_curve", "permutation_importance_ap"]
            },
            indent=2,
        )
    )
