"""Carga de los datos de Sephora (Kaggle: nadyinky/sephora-products-and-skincare-reviews).

Las reseñas vienen repartidas en varios CSV pesados (~500 MB). La primera vez se leen,
se limpian y se guardan en data/interim/reviews.parquet; las siguientes se lee el parquet (segundos).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "sephora"
INTERIM = ROOT / "data" / "interim"

REVIEW_COLS = [
    "author_id", "product_id", "rating", "is_recommended", "helpfulness",
    "total_feedback_count", "submission_time",
    "skin_tone", "skin_type", "eye_color", "hair_color",
    "review_text", "review_title",
]

# Orden de los tonos de piel de más claro a más oscuro (para gráficos y análisis por tono).
# "notSureST" y los valores raros se tratan aparte.
SKIN_TONE_ORDER = [
    "porcelain", "fair", "fairLight", "light", "lightMedium", "medium",
    "mediumTan", "olive", "tan", "deep", "rich", "dark", "ebony",
]


def load_products() -> pd.DataFrame:
    """Catálogo completo de productos (todas las categorías)."""
    return pd.read_csv(RAW / "product_info.csv")


def load_reviews(use_cache: bool = True, with_text: bool = False) -> pd.DataFrame:
    """Todas las reseñas en un solo DataFrame.

    with_text=False omite el texto de las reseñas (ahorra mucha memoria).
    """
    cache = INTERIM / "reviews.parquet"
    if use_cache and cache.exists():
        cols = None if with_text else [c for c in REVIEW_COLS if c not in ("review_text", "review_title")]
        return pd.read_parquet(cache, columns=cols)

    frames = [
        pd.read_csv(f, usecols=REVIEW_COLS, dtype={"author_id": str, "product_id": str}, low_memory=False)
        for f in sorted(RAW.glob("reviews_*.csv"))
    ]
    df = pd.concat(frames, ignore_index=True)

    df["submission_time"] = pd.to_datetime(df["submission_time"], errors="coerce")
    for c in ["skin_tone", "skin_type", "eye_color", "hair_color"]:
        df[c] = df[c].astype("category")

    n_before = len(df)
    df = df.drop_duplicates(subset=["author_id", "product_id", "submission_time", "rating", "review_text"])
    df.attrs["duplicates_removed"] = n_before - len(df)

    INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    if not with_text:
        df = df.drop(columns=["review_text", "review_title"])
    return df
