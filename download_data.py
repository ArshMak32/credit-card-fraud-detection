from pathlib import Path
from sklearn.datasets import fetch_openml

if __name__ == "__main__":
    print(
        "Downloading OpenML data ID 1597. This can take several minutes; raw data is not committed."
    )
    bunch = fetch_openml(data_id=1597, as_frame=True, parser="auto")
    df = bunch.data.copy()
    df["Class"] = bunch.target.astype(int)
    path = Path("data/creditcard.csv")
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    print(path)
