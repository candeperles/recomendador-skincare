"""Evaluación de recomendadores: la misma vara para todos los modelos.

Cada modelo implementa `score(user_ids) -> np.ndarray [n_personas, n_productos]`: un puntaje
por producto para cada persona (más alto = más recomendable). Esta función arma el top-k y mide.

Métricas (por persona, después se promedian):
- **Recall@k**: de los productos que le gustaron (y escondimos), qué proporción quedó en su top k.
  Si tiene más de k productos escondidos, el máximo posible es k, así que dividimos por min(n, k).
- **NDCG@k**: como el recall, pero premia más acertar arriba de la lista (posición 1 vale más que la 10).
- **Cobertura**: qué proporción del catálogo aparece en al menos un top k. Mide sesgo de popularidad.
"""
import numpy as np
import pandas as pd

from src.data import TONE_GROUP_ORDER


class ItemIndex:
    """Traduce product_id <-> posición en las matrices (0..n_productos-1)."""

    def __init__(self, items: pd.DataFrame):
        self.ids = items.product_id.to_numpy()
        self.pos = pd.Series(np.arange(len(self.ids)), index=self.ids)

    def __len__(self):
        return len(self.ids)

    def to_pos(self, product_ids) -> np.ndarray:
        return self.pos.reindex(product_ids).to_numpy()


def evaluate(model, eval_df: pd.DataFrame, train: pd.DataFrame, users: pd.DataFrame,
             idx: ItemIndex, k: int = 10, exclude_train: bool = True, batch: int = 4000) -> pd.DataFrame:
    """Devuelve una fila por persona con recall@k, ndcg@k, su grupo de tono y su top k.

    exclude_train=True: no se recomiendan productos que la persona ya reseñó en train
    (ya los conoce; recomendárselos sería trampa y no sirve).
    """
    truth = eval_df.assign(pos=idx.to_pos(eval_df.product_id)).groupby("author_id").pos.apply(np.array)
    user_ids = truth.index.to_numpy()
    tr = train[train.author_id.isin(user_ids)]
    seen = tr.assign(pos=idx.to_pos(tr.product_id)).groupby("author_id").pos.apply(np.array)

    discounts = 1 / np.log2(np.arange(2, k + 2))
    rows, tops = [], []
    for start in range(0, len(user_ids), batch):
        u = user_ids[start:start + batch]
        scores = model.score(u).astype(np.float32, copy=True)
        if exclude_train:
            for i, a in enumerate(u):
                if a in seen.index:
                    scores[i, seen[a]] = -np.inf
        # top-k sin ordenar todo el catálogo: argpartition y después ordenar solo esos k
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
        top = np.take_along_axis(top, order, axis=1)
        for i, a in enumerate(u):
            rel = truth[a]
            hit = np.isin(top[i], rel)
            n_ideal = min(len(rel), k)
            rows.append((a, len(rel), hit.sum() / n_ideal, (hit * discounts).sum() / discounts[:n_ideal].sum()))
        tops.append(top)

    res = pd.DataFrame(rows, columns=["author_id", "n_relevantes", "recall", "ndcg"]).set_index("author_id")
    res["tone_group"] = users.tone_group.reindex(res.index).fillna("desconocido")
    res["top"] = list(np.vstack(tops))
    return res


def summarize(res: pd.DataFrame, n_items: int, name: str = "") -> pd.Series:
    """Promedios con error estándar y cobertura de catálogo."""
    n = len(res)
    covered = np.unique(np.concatenate(res.top.to_numpy())).size
    return pd.Series({
        "recall@10": res.recall.mean(),
        "± recall": res.recall.std() / np.sqrt(n),
        "ndcg@10": res.ndcg.mean(),
        "± ndcg": res.ndcg.std() / np.sqrt(n),
        "cobertura": covered / n_items,
        "personas": n,
    }, name=name)


def by_tone(res: pd.DataFrame, metric: str = "recall") -> pd.DataFrame:
    """Métrica por grupo de tono, con error estándar."""
    g = res.groupby("tone_group")[metric].agg(["mean", "std", "size"])
    g["±"] = g["std"] / np.sqrt(g["size"])
    return g.reindex(TONE_GROUP_ORDER + ["desconocido"])[["mean", "±", "size"]]
