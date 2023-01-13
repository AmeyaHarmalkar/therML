from pathlib import Path
import pickle as pkl
import pandas as pd
import math
import numpy as np
from itertools import product
from collections import defaultdict


def process(name: str, df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    df["Method"] = "MCMC" if "mcmc" in name else "Top Mutants"
    df["Model"] = "ESM-1v Unsupervised" if "esm1v" in name else "ESM-1b Supervised"
    df = df.rename(columns={"rescore": "score", "likelihood": "score", "mutants": "mutant"})
    df = df.sort_values("score", ascending=False)
    if "mcmc" in name:
        window = math.ceil(window * len(df) / 10000)
        indices = np.array([], dtype=np.int64)
        for index in df.index:
            if len(indices) == 0 or np.min(np.abs(indices - index)) > window:
                indices = np.append(indices, index)
        df = df.loc[indices]
    return df.reset_index()


results = defaultdict(list)
for path in Path("multi-mutants").glob("*.pkl"):
    if path.stem == "final":
        continue
    with path.open("rb") as f:
        data = pkl.load(f)
        for key, df in data.items():
            df = process(path.stem, df)
            if isinstance(key, tuple):
                key = key[0]
            results[key].append(df)

for key, value in results.items():
    df = pd.concat(value)
    df = df.sort_values("score", ascending=False)
    for model, method in product(df["Model"].unique(), df["Method"].unique()):
        mask = (df["Model"] == model) & (df["Method"] == method)
        df.loc[mask, "index"] = np.arange(mask.sum(), dtype=np.int64)
    df["index"] = df["index"].astype(int)
    df = df.pivot(index="index", columns=["Model", "Method"], values="mutant")
    results[key] = df

with open("./multi-mutants/final.pkl", "wb") as f:
    pkl.dump(results, f)

for key, value in results.items():
    value.to_csv(f"./multi-mutants/{key}.csv", index=False)
