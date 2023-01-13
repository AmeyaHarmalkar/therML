from supervised import StabilityDataset, DATA_DIR, LOG_DIR
import pickle as pkl
import pandas as pd
import numpy as np
from scipy.stats import spearmanr


MODEL_NAME = "baseesm_headconcat_seed0_lr0.001_batch64"
LOG_DIR = LOG_DIR / MODEL_NAME
df = pd.read_csv(DATA_DIR / "ts50_all.csv")
trrosetta_entropy = pd.read_csv(DATA_DIR / "trrosetta_entropy.csv")
df = pd.merge(df, trrosetta_entropy, on="Name")
df["conf"] = df[[f"xa{i}_conf" for i in "abcde"]].mean(axis=1)
df["ent"] = df[[f"xa{i}_ent" for i in "abcde"]].mean(axis=1)

def get_correlations_single():
    NUM_SPLITS = 17
    metrics = []
    for split in range(NUM_SPLITS):
        data = StabilityDataset(split=split, mode="test")
        group = data.split
        log_dir = LOG_DIR / group
        with open(log_dir / "predictions.pkl", "rb") as f:
            predictions = pkl.load(f)["prediction"]
        probs = predictions.softmax(-1)

        p_70 = probs[:, -1]
        try:
            tr_conf = df.set_index("Name").loc[data.data["Name"], "conf"]
            tr_ent = df.set_index("Name").loc[data.data["Name"], "ent"]
        except KeyError:
            continue
        ts50_values = data.data["TS50_float"]

        metrics.append(
            {
                "Project": group,
                "Correlation": spearmanr(ts50_values, tr_conf).correlation,
                "Method": "TRRosetta Top",
            }
        )
        metrics.append(
            {
                "Project": group,
                "Correlation": spearmanr(ts50_values, tr_ent).correlation,
                "Method": "TRRosetta Entropy",
            }
        )
        metrics.append(
            {
                "Project": group,
                "Correlation": spearmanr(ts50_values, p_70).correlation,
                "Method": "ESM-1b Supervised",
            }
        )

    result = pd.DataFrame(metrics)
    return result