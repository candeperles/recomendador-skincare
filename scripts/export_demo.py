"""Exporta lo mínimo que necesita la demo para correr sin los datos de reseñas (por ejemplo, en la nube).

Escribe en web/assets/:
- items.parquet: el catálogo (2.351 productos) con lo que usa la torre producto y la demo,
  más la cantidad de reseñas positivas de cada producto (para la "variedad").
- perfil_categorias.json: los valores de tono, tipo de piel, ojos y pelo con los que se entrenó la red.
- two_tower_final.pt: los pesos del modelo final (scripts/train_final.py).

Datos derivados de "Sephora Products and Skincare Reviews" (Nady Inky, Kaggle), licencia CC BY 4.0.

Uso:  python scripts/export_demo.py
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data import PROFILE_COLS
from src.evaluate import ItemIndex
from src.splits import load_splits
from src.two_tower import DemoRecommender, TwoTowerRecommender

COLS = ["product_id", "product_name", "brand_name", "price_usd", "secondary_category", "tertiary_category",
        "ingredients", "highlights", "rating"]
KW = dict(dim=128, hidden=512)

if __name__ == "__main__":
    out = ROOT / "web" / "assets"
    out.mkdir(parents=True, exist_ok=True)
    s = load_splits()
    idx = ItemIndex(s.items)
    todo = pd.concat([s.train, s.val_warm, s.val_cold], ignore_index=True)

    items = s.items[COLS].reset_index(drop=True).copy()  # mismo orden que en el entrenamiento
    items["positivos"] = np.bincount(idx.to_pos(todo.product_id), minlength=len(idx))
    items.to_parquet(out / "items.parquet", index=False)

    cats = {c: list(pd.Categorical(s.users[c].astype(str)).categories) for c in PROFILE_COLS}
    (out / "perfil_categorias.json").write_text(json.dumps(cats, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.copy(ROOT / "models" / "two_tower_final.pt", out / "two_tower_final.pt")

    # chequeo: la versión liviana tiene que dar exactamente lo mismo que la completa
    full = TwoTowerRecommender(todo, s.users, s.items, idx, weights=ROOT / "models" / "two_tower_final.pt", **KW)
    lite = DemoRecommender(pd.read_parquet(out / "items.parquet"), cats, out / "two_tower_final.pt", **KW)
    perf = pd.DataFrame([{"skin_tone": t, "skin_type": k, "eye_color": "brown", "hair_color": "black"}
                         for t in ["fair", "deep", "desconocido"] for k in ["dry", "oily"]])
    hist = [[], [5], [], [10, 20], [], []]
    a, b = full.score_profiles(perf, hist), lite.score_profiles(perf, hist)
    assert np.allclose(a, b, atol=1e-5), np.abs(a - b).max()
    print("ok: la demo liviana coincide con el modelo completo")
    for f in sorted(out.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.2f} MB")
