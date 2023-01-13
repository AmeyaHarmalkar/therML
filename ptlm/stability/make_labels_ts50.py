import pandas as pd
import numpy as np
from typing import Union

xl = pd.ExcelFile("data/dataset_seq_TS50_v2.xlsx")

data = pd.concat([xl.parse(sheet).rename({"TS50": "TS50_raw", "TS 50": "TS50_raw", "TS50 (C)": "TS50_raw"}, axis="columns").assign(Sheet=sheet) for sheet in xl.sheet_names], 0).reset_index()


def to_float(x: Union[str, int]) -> float:
    try:
        return float(x)
    except ValueError:
        if x == "up":
            return 70
        else:
            return float("nan")


def digitize(ts50: float) -> int:
    if np.isnan(ts50):
        return -1
    else:
        return np.digitize(ts50, [50, 60, 70])

data["TS50_float"] = data["TS50_raw"].apply(to_float)
data["label"] = data["TS50_float"].apply(digitize)

# data.to_csv("data/ts50_all.csv", index=False)
