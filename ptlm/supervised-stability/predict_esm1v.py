import torch
import numpy as np
import pandas as pd
from evo.likelihood import pseudo_ppl
import esm
from pathlib import Path

DATA_FILE = Path(__file__).parents[1] / "data" / "all_data.csv"

model_name = [f"esm1v_t33_650M_UR90S_{i}" for i in range(1, 6)]
df = pd.read_csv(DATA_FILE)
sequences = df["sequence"]

torch.set_grad_enabled(False)


scores = []
for name in model_name:
    model, alphabet = esm.pretrained.load_model_and_alphabet_hub(name)
    model = model.eval().cuda().requires_grad_(False)
    result = pseudo_ppl(model, alphabet, sequences, mask_positions=False)
    scores.append(result)
scores = torch.stack(scores, 0).mean(0).numpy()

np.save("esm1v_preds.npy", scores)
