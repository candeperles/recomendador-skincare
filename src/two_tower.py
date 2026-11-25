"""Red two-tower: una torre para la persona y otra para el producto.

    torre persona:  perfil (tono, tipo de piel, ojos, pelo) + historial  ->  vector de `dim` números
    torre producto: ID, marca, categorías, precio, ingredientes, highlights  ->  vector de `dim` números
    puntaje = coseno(persona, producto) / temperatura

Decisiones (explicadas en notebooks/03_two_tower.ipynb):
- Softmax sobre TODO el catálogo: con 2.351 productos podemos comparar a cada persona contra
  todos en cada paso, así que no hace falta muestrear negativos.
- El historial de una persona es el promedio de los vectores (de la torre producto) de lo que le
  gustó, SIN el producto que estamos tratando de predecir (si no, sería hacer trampa).
- "History dropout": a veces le borramos el historial a propósito, para que la red también
  aprenda a recomendar solo con el perfil (cold start).
"""
import ast
import re
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import PROFILE_COLS
from src.evaluate import ItemIndex


# ---------------------------------------------------------------- features de producto

def _parse_list(s):
    """Las columnas vienen como texto con forma de lista de Python: "['a', 'b']"."""
    if not isinstance(s, str):
        return []
    try:
        out = ast.literal_eval(s)
        return out if isinstance(out, list) else [str(out)]
    except (ValueError, SyntaxError):
        return [s]


def parse_ingredients(s):
    """Texto de ingredientes -> lista normalizada ("Water (Aqua, Eau)" -> "water").

    Saca lo que está entre paréntesis, asteriscos y puntos, pasa a minúscula y separa por comas.
    Los productos que son kits traen encabezados ("Serum:"); quedan como ruido que después
    filtramos por frecuencia mínima.
    """
    text = " , ".join(_parse_list(s))
    text = re.sub(r"\([^)]*\)", " ", text)
    toks = []
    for t in re.split(r"[,;]", text):
        t = re.sub(r"[*.•]", " ", t.lower())
        t = re.sub(r"\s+", " ", t).strip(" -:")
        if 2 < len(t) < 50:
            toks.append(t)
    return list(dict.fromkeys(toks))  # sin repetidos, mantiene orden


def _vocab_bag(lists, min_count):
    """Listas de tokens -> (índices planos, offsets, vocabulario). El índice 0 = "ninguno"."""
    counts = pd.Series([t for l in lists for t in l]).value_counts()
    vocab = {t: i + 1 for i, t in enumerate(counts[counts >= min_count].index)}
    flat, offsets = [], []
    for l in lists:
        offsets.append(len(flat))
        ids = [vocab[t] for t in l if t in vocab]
        flat.extend(ids or [0])
    return np.array(flat), np.array(offsets), vocab


def _codes(series):
    """Categoría -> entero (0 = faltante)."""
    cats = pd.Categorical(series.fillna("__na__"))
    return cats.codes.astype(np.int64) + 1, len(cats.categories) + 1


@dataclass
class ItemFeatures:
    brand: torch.Tensor
    cat2: torch.Tensor
    cat3: torch.Tensor
    price: torch.Tensor
    ingr: torch.Tensor
    ingr_off: torch.Tensor
    high: torch.Tensor
    high_off: torch.Tensor
    sizes: dict = field(default_factory=dict)
    vocab_ingr: dict = field(default_factory=dict)
    vocab_high: dict = field(default_factory=dict)

    @classmethod
    def build(cls, items, min_ingr=5, min_high=5):
        brand, nb = _codes(items.brand_name)
        cat2, n2 = _codes(items.secondary_category)
        cat3, n3 = _codes(items.tertiary_category)
        p = np.log1p(items.price_usd.fillna(items.price_usd.median()).to_numpy())
        p = (p - p.mean()) / p.std()
        ingr, ingr_off, vi = _vocab_bag([parse_ingredients(s) for s in items.ingredients], min_ingr)
        high, high_off, vh = _vocab_bag([_parse_list(s) for s in items.highlights], min_high)
        t = lambda a: torch.as_tensor(a)
        return cls(t(brand), t(cat2), t(cat3), t(p).float().unsqueeze(1),
                   t(ingr), t(ingr_off), t(high), t(high_off),
                   sizes=dict(brand=nb, cat2=n2, cat3=n3, ingr=len(vi) + 1, high=len(vh) + 1),
                   vocab_ingr=vi, vocab_high=vh)


# ---------------------------------------------------------------- features de persona

class UserData:
    """Perfil codificado + historial (positivos de train) de cada persona, en arrays de numpy.

    El historial se guarda en formato CSR: los productos de la persona r están en
    hist[ptr[r]:ptr[r+1]].
    """

    def __init__(self, users: pd.DataFrame, train: pd.DataFrame, idx: ItemIndex, cols=PROFILE_COLS):
        self.cols = list(cols)
        self.row = pd.Series(np.arange(len(users)), index=users.index)
        self.profile, self.sizes, self.categories = [], [], []
        for c in self.cols:
            cats = pd.Categorical(users[c].astype(str))
            self.profile.append(cats.codes.astype(np.int64) + 1)
            self.sizes.append(len(cats.categories) + 1)
            self.categories.append(cats.categories)
        self.profile = np.stack(self.profile, axis=1)

        r = self.row[train.author_id].to_numpy()
        order = np.argsort(r, kind="stable")
        self.hist = idx.to_pos(train.product_id)[order]
        counts = np.bincount(r, minlength=len(users))
        self.ptr = np.concatenate([[0], np.cumsum(counts)])

    def encode_profiles(self, profiles: pd.DataFrame):
        """Perfiles nuevos (DataFrame con las columnas de perfil) -> enteros. Valor no visto -> 0."""
        return np.stack([cats.get_indexer(profiles[c].astype(str)) + 1
                         for c, cats in zip(self.cols, self.categories)], axis=1).astype(np.int64)

    def rows(self, author_ids):
        return self.row[np.asarray(author_ids)].to_numpy()

    def gather_history(self, rows):
        """Para un lote de personas: índices planos de su historial y cantidad por persona."""
        starts, lens = self.ptr[rows], self.ptr[rows + 1] - self.ptr[rows]
        total = lens.sum()
        ends = np.cumsum(lens)
        flat = np.repeat(starts - (ends - lens), lens) + np.arange(total)
        return self.hist[flat], lens


# ---------------------------------------------------------------- la red

def mlp(d_in, d_hidden, d_out, dropout):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_hidden, d_out))


class TwoTower(nn.Module):
    def __init__(self, n_items, item_sizes, profile_sizes, dim=64, d_feat=32, hidden=256, dropout=0.1,
                 temperature=0.05, use_profile=True, use_history=True, use_content=True, use_item_id=True):
        super().__init__()
        self.temperature = temperature
        self.use_profile, self.use_history = use_profile, use_history
        self.use_content, self.use_item_id = use_content, use_item_id

        # torre producto
        self.item_id = nn.Embedding(n_items, d_feat)
        self.brand = nn.Embedding(item_sizes["brand"], d_feat)
        self.cat2 = nn.Embedding(item_sizes["cat2"], d_feat // 2)
        self.cat3 = nn.Embedding(item_sizes["cat3"], d_feat // 2)
        self.ingr = nn.EmbeddingBag(item_sizes["ingr"], d_feat, mode="mean")
        self.high = nn.EmbeddingBag(item_sizes["high"], d_feat // 2, mode="mean")
        # id + marca + (cat2 + cat3) + ingredientes + highlights + precio
        d_item = d_feat + d_feat + d_feat + d_feat + d_feat // 2 + 1
        self.item_mlp = mlp(d_item, hidden, dim, dropout)

        # torre persona
        self.profile = nn.ModuleList([nn.Embedding(n, d_feat // 2) for n in profile_sizes])
        d_user = len(profile_sizes) * (d_feat // 2) + dim + 2  # perfil + historial + (tiene historial, log n)
        self.user_mlp = mlp(d_user, hidden, dim, dropout)

    def item_vectors(self, f: ItemFeatures):
        n = f.brand.shape[0]
        z = lambda d: torch.zeros(n, d)
        idv = self.item_id.weight if self.use_item_id else z(self.item_id.embedding_dim)
        if self.use_content:
            parts = [self.brand(f.brand), self.cat2(f.cat2), self.cat3(f.cat3),
                     self.ingr(f.ingr, f.ingr_off), self.high(f.high, f.high_off), f.price]
        else:
            parts = [z(self.brand.embedding_dim), z(self.cat2.embedding_dim), z(self.cat3.embedding_dim),
                     z(self.ingr.embedding_dim), z(self.high.embedding_dim), z(1)]
        return F.normalize(self.item_mlp(torch.cat([idv] + parts, dim=1)), dim=1)

    def user_vectors(self, profile, hist_sum, hist_n):
        """profile: [B, n_cols] enteros; hist_sum: [B, dim] suma de vectores del historial; hist_n: [B]."""
        B = profile.shape[0]
        if self.use_profile:
            prof = [emb(profile[:, j]) for j, emb in enumerate(self.profile)]
        else:
            prof = [torch.zeros(B, emb.embedding_dim) for emb in self.profile]
        n = hist_n.float().unsqueeze(1)
        if not self.use_history:
            hist_sum, n = torch.zeros_like(hist_sum), torch.zeros_like(n)
        hist = hist_sum / n.clamp(min=1)
        extra = torch.cat([(n > 0).float(), torch.log1p(n)], dim=1)
        return F.normalize(self.user_mlp(torch.cat(prof + [hist, extra], dim=1)), dim=1)


# ---------------------------------------------------------------- envoltorio: entrenar y puntuar

def _history_sum(item_vecs, hist_items, lens):
    """Suma, por persona, de los vectores de los productos de su historial."""
    B = len(lens)
    if len(hist_items) == 0:
        return torch.zeros(B, item_vecs.shape[1])
    seg = torch.as_tensor(np.repeat(np.arange(B), lens))
    out = torch.zeros(B, item_vecs.shape[1])
    return out.index_add(0, seg, item_vecs[torch.as_tensor(hist_items)])


class TwoTowerRecommender:
    """Entrena la red y expone `score(user_ids)` como los baselines.

    Con `weights=<ruta .pt>` no entrena: arma la red y carga pesos ya entrenados
    (hay que pasar los mismos `model_kw` con los que se entrenó).
    """

    def __init__(self, train, users, items, idx: ItemIndex, epochs=8, batch_size=1024, lr=2e-3,
                 weight_decay=1e-5, history_dropout=0.3, seed=42, threads=8, verbose=True,
                 eval_fn=None, weights=None, **model_kw):
        if weights is not None:
            epochs = 0
        torch.manual_seed(seed)
        torch.set_num_threads(threads)
        self.rng = np.random.default_rng(seed)
        self.idx, self.verbose = idx, verbose
        self.fi = ItemFeatures.build(items)
        self.ud = UserData(users, train, idx)
        self.model = TwoTower(len(idx), self.fi.sizes, self.ud.sizes, **model_kw)
        self.history_dropout = history_dropout

        self.log = []
        if weights is not None:
            self.model.load_state_dict(torch.load(weights))
            self.log = pd.DataFrame(self.log)
            return

        rows = self.ud.rows(train.author_id)
        targets = idx.to_pos(train.product_id)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        steps = epochs * int(np.ceil(len(rows) / batch_size))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)

        best, self.best_state = -np.inf, None
        for ep in range(1, epochs + 1):
            t0 = time.time()
            self.model.train()
            losses = []
            perm = self.rng.permutation(len(rows))
            for s in range(0, len(perm), batch_size):
                b = perm[s:s + batch_size]
                loss = self._loss(rows[b], targets[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                losses.append(loss.item())
            entry = {"epoch": ep, "loss": float(np.mean(losses)), "seg": time.time() - t0}
            if eval_fn is not None:
                entry.update(eval_fn(self))
                if entry["criterio"] > best:
                    best = entry["criterio"]
                    self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.log.append(entry)
            if verbose:
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in entry.items()})
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self.log = pd.DataFrame(self.log)
        self._cache = None

    def _loss(self, rows, targets):
        item_vecs = self.model.item_vectors(self.fi)
        hist_items, lens = self.ud.gather_history(rows)
        hist_sum = _history_sum(item_vecs, hist_items, lens)
        t = torch.as_tensor(targets)
        # sacar el producto objetivo de su propio historial (siempre está, porque viene de train)
        hist_sum = hist_sum - item_vecs[t]
        n = torch.as_tensor(lens - 1)
        # history dropout: a algunas personas les borramos el historial para practicar cold start
        drop = torch.as_tensor(self.rng.random(len(rows)) < self.history_dropout)
        hist_sum = torch.where(drop.unsqueeze(1), torch.zeros_like(hist_sum), hist_sum)
        n = torch.where(drop, torch.zeros_like(n), n)
        u = self.model.user_vectors(torch.as_tensor(self.ud.profile[rows]), hist_sum, n)
        logits = u @ item_vecs.T / self.model.temperature
        return F.cross_entropy(logits, t)

    @torch.no_grad()
    def item_vectors(self):
        self.model.eval()
        return self.model.item_vectors(self.fi)

    @torch.no_grad()
    def user_vectors(self, user_ids, item_vecs=None):
        self.model.eval()
        item_vecs = self.item_vectors() if item_vecs is None else item_vecs
        rows = self.ud.rows(user_ids)
        hist_items, lens = self.ud.gather_history(rows)
        hist_sum = _history_sum(item_vecs, hist_items, lens)
        return self.model.user_vectors(torch.as_tensor(self.ud.profile[rows]), hist_sum, torch.as_tensor(lens))

    @torch.no_grad()
    def score(self, user_ids):
        V = self.item_vectors()
        return (self.user_vectors(user_ids, V) @ V.T).numpy()

    @torch.no_grad()
    def score_profiles(self, profiles: pd.DataFrame, histories=None):
        """Puntajes para personas inventadas.

        profiles: DataFrame con las columnas de perfil (una fila por persona).
        histories: opcional, una lista (una por persona) de posiciones de productos que le gustaron.
                   Sin historial es cold start puro. La red no se re-entrena: el historial es una entrada.
        """
        return _score_profiles(self.model, self.fi, self.ud.encode_profiles(profiles), histories)


@torch.no_grad()
def _score_profiles(model, fi, prof_codes, histories=None):
    """Puntaje (coseno) de cada producto para perfiles ya codificados, con historial opcional."""
    model.eval()
    V = model.item_vectors(fi)
    B = len(prof_codes)
    histories = histories if histories is not None else [[] for _ in range(B)]
    lens = np.array([len(h) for h in histories])
    flat = np.concatenate([np.asarray(h, dtype=np.int64) for h in histories]) if lens.sum() else np.array([], dtype=np.int64)
    hist_sum = _history_sum(V, flat, lens)
    u = model.user_vectors(torch.as_tensor(prof_codes), hist_sum, torch.as_tensor(lens))
    return (u @ V.T).numpy()


class DemoRecommender:
    """Versión liviana para la demo: no necesita los datos de reseñas, solo
    (1) el catálogo de productos, (2) las categorías de perfil con las que se entrenó y (3) los pesos.
    Los genera scripts/export_demo.py en web/assets/.
    """

    def __init__(self, items: pd.DataFrame, categories: dict, weights, **model_kw):
        self.fi = ItemFeatures.build(items)
        self.cols = list(categories)
        self.categories = [pd.Index(v) for v in categories.values()]
        sizes = [len(c) + 1 for c in self.categories]
        self.model = TwoTower(len(items), self.fi.sizes, sizes, **model_kw)
        self.model.load_state_dict(torch.load(weights))
        self.model.eval()

    def encode_profiles(self, profiles: pd.DataFrame):
        return np.stack([cats.get_indexer(profiles[c].astype(str)) + 1
                         for c, cats in zip(self.cols, self.categories)], axis=1).astype(np.int64)

    def score_profiles(self, profiles: pd.DataFrame, histories=None):
        return _score_profiles(self.model, self.fi, self.encode_profiles(profiles), histories)
