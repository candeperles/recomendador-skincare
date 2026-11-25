"""Entrena la two-tower final para la evaluación en test.

Protocolo estándar: una vez elegidos los hiperparámetros en validación, se re-entrena el modelo
con TODO lo que no es test (train + validación con historial + validación cold start).
Sin validación para elegir la época, usamos las 10 épocas completas (en el notebook 03 la
época 10 rindió prácticamente igual que la mejor).

Uso:  python scripts/train_final.py      -> models/two_tower_final.pt  (~20 min en CPU)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from src.evaluate import ItemIndex
from src.splits import load_splits
from src.two_tower import TwoTowerRecommender

if __name__ == "__main__":
    s = load_splits()
    idx = ItemIndex(s.items)
    todo_menos_test = pd.concat([s.train, s.val_warm, s.val_cold], ignore_index=True)
    with threadpool_limits(1, "blas"):
        m = TwoTowerRecommender(todo_menos_test, s.users, s.items, idx, epochs=10, threads=12,
                                dim=128, hidden=512)
    torch.save(m.model.state_dict(), ROOT / "models" / "two_tower_final.pt")
    print(m.log.to_string())
