"""Entrena las variantes de la two-tower para el paso 4 (ablaciones y semillas).

Cada variante usa la configuración final del paso 3 (dim 128, hidden 512, 10 épocas, mejor
época según validación) y cambia UNA sola cosa. Guarda los pesos en models/ablaciones/<nombre>.pt
y el registro por época en models/ablaciones/<nombre>_log.csv.

Uso:  python scripts/train_ablations.py sin_perfil sin_contenido ... [threads=8]   (sin nombres: todas)
Tarda ~20-25 min por variante en CPU.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from grid_two_tower import make_eval_fn
from src.evaluate import ItemIndex
from src.splits import load_splits
from src.two_tower import TwoTowerRecommender

BASE = dict(dim=128, hidden=512, epochs=10)
VARIANTES = {
    "sin_perfil": dict(use_profile=False),        # la persona es solo su historial
    "sin_historial": dict(use_history=False),     # la persona es solo su perfil
    "sin_contenido": dict(use_content=False),     # el producto es solo su ID (como en ALS)
    "sin_id": dict(use_item_id=False),            # el producto es solo su contenido (sirve para productos nuevos)
    "semilla_43": dict(seed=43),                  # mismo modelo, otra semilla: ¿qué tan estable es?
    "semilla_44": dict(seed=44),
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(VARIANTES)
    threads = int(next((a.split("=")[1] for a in names if a.startswith("threads=")), 8))
    names = [n for n in names if not n.startswith("threads=")]
    s = load_splits()
    idx = ItemIndex(s.items)
    eval_fn = make_eval_fn(s, idx)
    out = ROOT / "models" / "ablaciones"
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        print("==", name, flush=True)
        with threadpool_limits(1, "blas"):
            m = TwoTowerRecommender(s.train, s.users, s.items, idx, eval_fn=eval_fn, threads=threads,
                                    verbose=False, **BASE, **VARIANTES[name])
        torch.save(m.model.state_dict(), out / f"{name}.pt")
        m.log.assign(variante=name).to_csv(out / f"{name}_log.csv", index=False)
        print(m.log.tail(3).to_string(), flush=True)
