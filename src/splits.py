"""Splits de entrenamiento, validación y test para el recomendador.

Hay dos escenarios de evaluación:

1. **Con historial (warm).** Personas que el modelo conoce. Para cada una escondemos sus
   últimos productos favoritos y vemos si el modelo los recupera a partir del resto.
2. **Cold start.** Personas que el modelo nunca vio. Solo conocemos su perfil de piel y
   queremos adivinar qué productos les van a gustar.

Reglas (decididas en el notebook 00):
- Positivo = rating >= 4.
- Agrupamos los positivos de cada persona por **día**: el 35% de las personas escribe todas
  sus reseñas el mismo día, así que "la última reseña" no está bien definida.
- Persona evaluable con historial = tiene positivos en al menos 3 días distintos.
  · test: un producto al azar del último día
  · validación: un producto al azar del anteúltimo día
  · los demás productos de esos dos días se descartan (para que el modelo no vea productos
    de la misma "sesión" que estamos tratando de adivinar)
  · train: todos los días anteriores
- Cold start: un 10% de las personas, elegido al azar, queda fuera del entrenamiento.
  La mitad es validación y la otra mitad test. Todos sus positivos son la respuesta correcta.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data import ROOT, user_profiles

PROCESSED = ROOT / "data" / "processed"
SEED = 42
MIN_DAYS = 3          # días distintos con positivos para evaluar a una persona con historial
COLD_FRACTION = 0.10  # proporción de personas reservada para cold start


@dataclass
class Splits:
    train: pd.DataFrame       # positivos para entrenar (personas con historial)
    val_warm: pd.DataFrame    # 1 producto por persona evaluable
    test_warm: pd.DataFrame   # 1 producto por persona evaluable
    val_cold: pd.DataFrame    # todos los positivos de personas cold start (validación)
    test_cold: pd.DataFrame   # todos los positivos de personas cold start (test)
    users: pd.DataFrame       # perfil de todas las personas + a qué grupo pertenecen
    items: pd.DataFrame       # catálogo de productos de skincare reseñados


def positives(reviews: pd.DataFrame) -> pd.DataFrame:
    """Reseñas con rating >= 4, una por persona y producto (si hay repetidas, la primera)."""
    pos = reviews.loc[reviews.rating >= 4, ["author_id", "product_id", "rating", "submission_time"]]
    pos = pos.sort_values("submission_time").drop_duplicates(["author_id", "product_id"], keep="first")
    pos["day"] = pos.submission_time.dt.normalize()
    return pos.reset_index(drop=True)


def _pick_one_per_user(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Elige una fila al azar por persona (con semilla fija, reproducible)."""
    shuffled = df.assign(_r=rng.random(len(df))).sort_values(["author_id", "_r"])
    return shuffled.drop_duplicates("author_id").drop(columns="_r")


def make_splits(reviews: pd.DataFrame, products: pd.DataFrame, seed: int = SEED) -> Splits:
    rng = np.random.default_rng(seed)
    pos = positives(reviews)
    users = user_profiles(reviews)

    # 1. Cold start: separar un 10% de las personas que tienen algún positivo
    all_pos_users = pos.author_id.unique()
    all_pos_users.sort()  # orden fijo antes de muestrear, para que sea reproducible
    cold = rng.choice(all_pos_users, size=int(len(all_pos_users) * COLD_FRACTION), replace=False)
    cold_val_users, cold_test_users = np.array_split(rng.permutation(cold), 2)

    users["grupo"] = "solo_train"
    users.loc[users.index.isin(cold_val_users), "grupo"] = "cold_val"
    users.loc[users.index.isin(cold_test_users), "grupo"] = "cold_test"
    users.loc[~users.index.isin(all_pos_users), "grupo"] = "sin_positivos"

    val_cold = pos[pos.author_id.isin(cold_val_users)].copy()
    test_cold = pos[pos.author_id.isin(cold_test_users)].copy()
    warm = pos[~pos.author_id.isin(cold)].copy()

    # 2. Con historial: ordenar los días de cada persona del último al primero
    days = warm[["author_id", "day"]].drop_duplicates()
    days["day_rank"] = days.groupby("author_id").day.rank(method="dense", ascending=False).astype(int)
    n_days = days.groupby("author_id").size()
    warm = warm.merge(days, on=["author_id", "day"])

    evaluable = n_days[n_days >= MIN_DAYS].index
    is_eval = warm.author_id.isin(evaluable)
    users.loc[users.index.isin(evaluable), "grupo"] = "warm_eval"

    test_warm = _pick_one_per_user(warm[is_eval & (warm.day_rank == 1)], rng)
    val_warm = _pick_one_per_user(warm[is_eval & (warm.day_rank == 2)], rng)

    # train: las personas no evaluables completas + las evaluables sin sus dos últimos días
    train = warm[~is_eval | (warm.day_rank >= 3)].copy()

    cols = ["author_id", "product_id", "rating", "submission_time"]
    # Catálogo = todos los productos con alguna reseña (aunque no tengan positivos)
    items = products[products.product_id.isin(reviews.product_id.unique())]
    items = items[["product_id", "product_name", "brand_name", "price_usd", "size",
                   "secondary_category", "tertiary_category", "ingredients", "highlights",
                   "loves_count", "rating", "reviews"]].reset_index(drop=True)

    return Splits(
        train=train[cols].reset_index(drop=True),
        val_warm=val_warm[cols].reset_index(drop=True),
        test_warm=test_warm[cols].reset_index(drop=True),
        val_cold=val_cold[cols].reset_index(drop=True),
        test_cold=test_cold[cols].reset_index(drop=True),
        users=users,
        items=items,
    )


def save_splits(s: Splits) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name in ["train", "val_warm", "test_warm", "val_cold", "test_cold", "items"]:
        getattr(s, name).to_parquet(PROCESSED / f"{name}.parquet", index=False)
    s.users.to_parquet(PROCESSED / "users.parquet")


def load_splits() -> Splits:
    """Lee los splits guardados en data/processed (los genera el notebook 01)."""
    read = lambda n: pd.read_parquet(PROCESSED / f"{n}.parquet")
    return Splits(
        train=read("train"), val_warm=read("val_warm"), test_warm=read("test_warm"),
        val_cold=read("val_cold"), test_cold=read("test_cold"),
        users=read("users"), items=read("items"),
    )
