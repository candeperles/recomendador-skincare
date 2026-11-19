"""Grilla de hiperparámetros de la two-tower (validación). Tarda ~20 min por configuración en CPU.

Uso:  python scripts/grid_two_tower.py
Escribe reports/03_two_tower_grid_val.csv (una fila por configuración, mejor época).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data import PROFILE_COLS
from src.evaluate import ItemIndex, evaluate
from src.splits import load_splits
from src.two_tower import TwoTowerRecommender

GRID = {
    "A · base (temp 0,05, hist. dropout 0,3)": {},
    "B · temperatura 0,1": {"temperature": 0.1},
    "C · hist. dropout 0,5": {"history_dropout": 0.5},
    "D · sin hist. dropout": {"history_dropout": 0.0},
    "E · red más grande (dim 128, hidden 512)": {"dim": 128, "hidden": 512},
    "F · temperatura 0,02": {"temperature": 0.02},
}


def make_eval_fn(s, idx):
    completo = s.users.index[(s.users[PROFILE_COLS] != "desconocido").all(axis=1)]

    def eval_fn(m):
        rw = evaluate(m, s.val_warm, s.train, s.users, idx)
        rc = evaluate(m, s.val_cold, s.train, s.users, idx, exclude_train=False)
        rcc = rc[rc.index.isin(completo)]
        out = {"warm": rw.recall.mean(), "cold": rc.recall.mean(), "cold_completo": rcc.recall.mean(),
               "cob_cold": np.unique(np.concatenate(rcc.top.to_numpy())).size / len(idx)}
        # criterio para elegir la mejor época: promedio de los dos escenarios principales
        out["criterio"] = (out["warm"] + out["cold_completo"]) / 2
        return out

    return eval_fn


if __name__ == "__main__":
    s = load_splits()
    idx = ItemIndex(s.items)
    eval_fn = make_eval_fn(s, idx)
    rows = []
    for name, kw in GRID.items():
        print("==", name)
        with threadpool_limits(1, "blas"):
            m = TwoTowerRecommender(s.train, s.users, s.items, idx, epochs=10, eval_fn=eval_fn, **kw)
        best = m.log.loc[m.log.criterio.idxmax()]
        rows.append({"configuración": name, "mejor época": int(best.epoch), "recall warm": best.warm,
                     "recall cold (perfil completo)": best.cold_completo, "cobertura cold": best.cob_cold})
    pd.DataFrame(rows).to_csv(ROOT / "reports" / "03_two_tower_grid_val.csv", index=False)
