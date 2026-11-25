"""Baselines: los rivales que la red two-tower tiene que vencer.

Todos implementan `score(user_ids) -> np.ndarray [n_personas, n_productos]`.

1. Popularity:         lo más popular en train, igual para todo el mundo.
2. SegmentPopularity:  lo más popular entre las personas con tu mismo tono y tipo de piel.
3. ALSRecommender:     factorización de matrices (solo sirve para personas con historial).
4. ProfileToALS:       híbrido para cold start: predice, desde el perfil, dónde cae la persona
                       en el "mapa de gustos" del ALS.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from src.data import PROFILE_COLS
from src.evaluate import ItemIndex


class Popularity:
    """Puntaje = cantidad de positivos en train (en log, para que sea una escala razonable)."""

    def __init__(self, train: pd.DataFrame, idx: ItemIndex):
        counts = np.bincount(idx.to_pos(train.product_id), minlength=len(idx))
        self.scores = np.log1p(counts).astype(np.float32)

    def score(self, user_ids):
        return np.tile(self.scores, (len(user_ids), 1))


class SegmentPopularity:
    """Popularidad dentro del segmento de la persona (por defecto, tono × tipo de piel).

    Los segmentos chicos tienen pocos datos y su ranking sería ruidoso. Por eso mezclamos con
    la popularidad global (*smoothing*): es como sumar `alpha` positivos "imaginarios"
    repartidos como en la población general.

        puntaje(producto) = (positivos_en_segmento + alpha * proporción_global) / (total_segmento + alpha)

    alpha chico → confía en el segmento; alpha grande → se parece a la popularidad global.
    """

    def __init__(self, train, users, idx: ItemIndex, cols=("skin_tone", "skin_type"), alpha=100.0):
        self.users, self.cols, self.n = users, list(cols), len(idx)
        seg = self._segment(train.author_id)
        pos = idx.to_pos(train.product_id)
        global_counts = np.bincount(pos, minlength=self.n).astype(np.float64)
        global_share = global_counts / global_counts.sum()

        df = pd.DataFrame({"seg": seg, "pos": pos})
        self.table = {}
        for s, g in df.groupby("seg"):
            c = np.bincount(g.pos, minlength=self.n)
            self.table[s] = ((c + alpha * global_share) / (c.sum() + alpha)).astype(np.float32)
        self.fallback = global_share.astype(np.float32)

    def _segment(self, author_ids):
        prof = self.users.loc[np.asarray(author_ids), self.cols].astype(str)
        seg = prof[self.cols[0]]
        for c in self.cols[1:]:
            seg = seg + "|" + prof[c]
        return seg.to_numpy()

    def score(self, user_ids):
        return np.vstack([self.table.get(s, self.fallback) for s in self._segment(user_ids)])


def interaction_matrix(train, idx: ItemIndex):
    """Matriz dispersa personas × productos con 1 donde hubo un positivo."""
    user_ids = np.sort(train.author_id.unique())
    upos = pd.Series(np.arange(len(user_ids)), index=user_ids)
    m = sp.csr_matrix(
        (np.ones(len(train), dtype=np.float32), (upos[train.author_id].to_numpy(), idx.to_pos(train.product_id))),
        shape=(len(user_ids), len(idx)),
    )
    m.data[:] = 1.0  # por si hubiera duplicados
    return m, upos


class ALSRecommender:
    """Factorización de matrices con ALS (Alternating Least Squares) para feedback implícito.

    Aprende un vector de `factors` números para cada persona y para cada producto, de forma que
    persona · producto sea alto si la persona tuvo un positivo con ese producto.
    `alpha` controla cuánto más pesa un positivo observado frente a un "no observado".
    Solo puede puntuar personas que vio en entrenamiento (no sirve para cold start).
    """

    def __init__(self, train, idx: ItemIndex, factors=64, regularization=0.05, alpha=10.0, iterations=20, seed=42):
        from implicit.als import AlternatingLeastSquares

        self.m, self.upos = interaction_matrix(train, idx)
        self.model = AlternatingLeastSquares(
            factors=factors, regularization=regularization, alpha=alpha,
            iterations=iterations, random_state=seed, use_gpu=False,
        )
        self.model.fit(self.m, show_progress=False)
        self.U = np.asarray(self.model.user_factors)
        self.V = np.asarray(self.model.item_factors)

    def score(self, user_ids):
        return self.U[self.upos[np.asarray(user_ids)].to_numpy()] @ self.V.T


class ProfileToALS:
    """Híbrido para cold start.

    1. Tomamos los vectores de persona que aprendió el ALS (su "lugar en el mapa de gustos").
    2. Entrenamos una regresión Ridge que predice ese vector a partir del perfil
       (tono, tipo de piel, ojos, pelo, en one-hot).
    3. Para una persona nueva: perfil → vector predicho → puntaje con los vectores de producto.

    Solo usamos personas con al menos `min_pos` positivos para la regresión: sus vectores
    están mejor estimados.
    """

    def __init__(self, als: ALSRecommender, train, users, min_pos=2, ridge_alpha=1.0, cols=PROFILE_COLS):
        self.als, self.users, self.cols = als, users, list(cols)
        counts = train.author_id.value_counts()
        fit_users = counts[counts >= min_pos].index
        self.enc = OneHotEncoder(handle_unknown="ignore")
        X = self.enc.fit_transform(users.loc[fit_users, self.cols].astype(str))
        Y = als.U[als.upos[fit_users].to_numpy()]
        self.reg = Ridge(alpha=ridge_alpha).fit(X, Y)

    def score(self, user_ids):
        X = self.enc.transform(self.users.loc[np.asarray(user_ids), self.cols].astype(str))
        return self.reg.predict(X) @ self.als.V.T


class PopularityPenalty:
    """Re-ranking: le resta a los puntajes de cualquier modelo una penalización por popularidad.

        puntaje_nuevo = scale * puntaje_modelo - beta * log(1 + positivos_en_train)

    beta = 0 deja el modelo igual; beta más grande empuja hacia productos menos conocidos.
    `scale` lleva el puntaje del modelo a una escala comparable (para la two-tower, 1/temperatura,
    que es la escala con la que se entrenó la softmax). No re-entrena nada.
    """

    def __init__(self, model, train, idx: ItemIndex, beta=0.5, scale=1.0):
        self.model, self.beta, self.scale = model, beta, scale
        self.logpop = np.log1p(np.bincount(idx.to_pos(train.product_id), minlength=len(idx))).astype(np.float32)

    def score(self, user_ids):
        return self.scale * self.model.score(user_ids) - self.beta * self.logpop
