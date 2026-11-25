"""Demo: elegís tu piel y la red two-tower te recomienda skincare.

Correr desde la raíz del proyecto:
    .venv\\Scripts\\streamlit run web/app.py

Usa el modelo final (models/two_tower_final.pt, ver scripts/train_final.py) y los datos procesados.
El tema (negro + rosa) está en .streamlit/config.toml y en el CSS de abajo.
"""
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src.evaluate import ItemIndex
from src.splits import load_splits
from src.two_tower import TwoTowerRecommender, _parse_list

st.set_page_config(page_title="Tu top 10 de skincare", page_icon="🧴", layout="wide")

TONOS = {
    "porcelain": "porcelain · muy clara", "fair": "fair · clara", "fairLight": "fairLight · clara",
    "light": "light · clara/media", "lightMedium": "lightMedium · media clara", "medium": "medium · media",
    "mediumTan": "mediumTan · media trigueña", "olive": "olive · oliva", "tan": "tan · trigueña",
    "deep": "deep · oscura", "rich": "rich · oscura", "dark": "dark · muy oscura", "ebony": "ebony · muy oscura",
}
TIPOS = {"dry": "seca", "oily": "grasa", "combination": "mixta", "normal": "normal"}
OJOS = {"brown": "marrones", "blue": "azules", "green": "verdes", "hazel": "avellana", "gray": "grises"}
PELO = {"brown": "castaño", "black": "negro", "blonde": "rubio", "red": "pelirrojo", "auburn": "caoba", "gray": "canoso"}
NO_SE = "desconocido"
TONOS_SESGADOS = {"dark", "porcelain", "olive"}
VARIEDAD = {0.0: "lo más probable", 0.25: "un poco", 0.5: "bastante", 0.75: "mucha"}

ROSA, NEGRO, CREMA = "#EACBC0", "#2B1E19", "#F7EFE9"
CARAMELO = "#C4977F"

# ---------------------------------------------------------------- estilo
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Anton&family=Inter:wght@400;500;600;700&display=swap');

/* Inter para el texto (sin tocar los íconos de Streamlit, que usan su propia tipografía) */
html, body, .stApp, .stMarkdown, p, label, input, button, li {{ font-family: 'Inter', sans-serif; }}
[data-testid="stIconMaterial"] {{ font-family: 'Material Symbols Rounded' !important; }}
/* Streamlit envuelve cada título en un <span>: que herede la tipografía del título */
h1 span, h2 span, h3 span {{ font-family: inherit !important; }}
[data-testid="stHeaderActionElements"] {{ display: none !important; }}
.block-container {{ padding-top: 2.2rem; max-width: 1100px; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* barra lateral rosa, como el panel derecho de la referencia */
section[data-testid="stSidebar"] {{ background: {ROSA}; }}
section[data-testid="stSidebar"] * {{ color: {NEGRO} !important; }}
section[data-testid="stSidebar"] h2 {{
    font-family: 'Anton', sans-serif; font-weight: 400; text-transform: uppercase;
    font-size: 1.9rem; line-height: 1; letter-spacing: .5px; margin: 1.2rem 0 .4rem 0;
}}
section[data-testid="stSidebar"] label p {{ font-weight: 600; font-size: .8rem; text-transform: uppercase; letter-spacing: .6px; }}
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="input"] > div {{
    background: {NEGRO} !important; border-radius: 999px !important; border: none !important;
}}
section[data-testid="stSidebar"] [data-baseweb="select"] * {{ color: {CREMA} !important; }}
section[data-testid="stSidebar"] [data-baseweb="select"] svg {{ fill: {ROSA} !important; }}
section[data-testid="stSidebar"] [data-baseweb="tag"] {{ background: {ROSA} !important; border-radius: 999px !important; }}
section[data-testid="stSidebar"] [data-baseweb="tag"] * {{ color: {NEGRO} !important; }}
section[data-testid="stSidebar"] [data-testid="stSliderThumbValue"],
section[data-testid="stSidebar"] [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] [data-testid="stTickBarMax"] {{ font-weight: 600; }}
/* sliders: barra, círculo y valor en marrón oscuro sobre el panel rosa */
section[data-testid="stSidebar"] [data-testid="stSlider"] [role="group"] {{ filter: brightness(0.28) sepia(0.7); }}
/* selectores (combobox con <input>): píldora negra con texto crema */
section[data-testid="stSidebar"] :is([data-testid="stSelectbox"], [data-testid="stMultiSelect"]) [role="group"] {{
    background: {NEGRO} !important; border-radius: 999px !important; border: none !important; padding-left: .4rem;
}}
section[data-testid="stSidebar"] :is([data-testid="stSelectbox"], [data-testid="stMultiSelect"]) [role="group"] :is(input, span, div) {{
    color: {CREMA} !important; -webkit-text-fill-color: {CREMA} !important;
}}
section[data-testid="stSidebar"] [role="group"] input::placeholder {{ color: #A8958B !important; -webkit-text-fill-color: #A8958B !important; }}
section[data-testid="stSidebar"] [role="group"] svg {{ fill: {ROSA} !important; color: {ROSA} !important; }}

/* encabezado */
.meta {{ display: flex; gap: 2.5rem; flex-wrap: wrap; margin-bottom: 1.6rem; }}
.meta div {{ font-size: .68rem; text-transform: uppercase; letter-spacing: .8px; color: #A8958B; line-height: 1.35; }}
.meta b {{ display: block; color: {CREMA}; font-size: .85rem; letter-spacing: .4px; }}
.hero {{ position: relative; margin-bottom: 1.4rem; }}
.hero h1 {{
    font-family: 'Anton', sans-serif; font-weight: 400; color: {ROSA}; text-transform: uppercase;
    font-size: clamp(3.2rem, 8.5vw, 7.5rem); line-height: .88; margin: 0; padding: 0; letter-spacing: 1px;
}}
.pills {{ display: flex; gap: .6rem; flex-wrap: wrap; margin-top: 1.5rem; }}
.pill {{
    display: inline-block; padding: .45rem 1.1rem; border-radius: 999px; font-weight: 700;
    font-size: .78rem; text-transform: uppercase; letter-spacing: .6px; line-height: 1.1;
}}
.pill.blanca {{ background: {CREMA}; color: {NEGRO}; transform: rotate(-6deg); }}
.pill.rosa {{ background: {ROSA}; color: {NEGRO}; transform: rotate(5deg); }}
.pill.borde {{ border: 1.5px solid {CARAMELO}; color: {CARAMELO}; transform: rotate(-2deg); }}
.sub {{ color: #C7B8AF; font-size: .95rem; max-width: 640px; margin: .4rem 0 1.4rem 0; }}

/* avisos */
.aviso {{ background: {ROSA}; color: {NEGRO}; border-radius: 22px; padding: 1rem 1.3rem; margin-bottom: 1.2rem; font-size: .92rem; }}
.aviso b {{ font-family: 'Anton', sans-serif; font-weight: 400; font-size: 1.15rem; text-transform: uppercase; display: block; margin-bottom: .2rem; }}
.nota {{ border: 1.5px solid #4A3A33; border-radius: 22px; padding: 1rem 1.3rem; margin-bottom: 1.2rem; color: #DCCFC7; font-size: .92rem; }}

/* tarjetas */
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 14px; }}
.card {{
    background: #271D19; border: 1px solid #3A2C26; border-radius: 26px; padding: 1.1rem 1.2rem 1rem 1.2rem;
    display: flex; gap: 1rem; align-items: flex-start; transition: border-color .15s ease;
}}
.card:hover {{ border-color: {CARAMELO}; }}
.card.top {{ background: {ROSA}; border-color: {ROSA}; }}
.card.top * {{ color: {NEGRO} !important; }}
.card.top .chip {{ border-color: {NEGRO} !important; }}
.card.top .precio {{ background: {NEGRO}; color: {ROSA} !important; }}
.rank {{ font-family: 'Anton', sans-serif; color: {CARAMELO}; font-size: 3rem; line-height: .9; min-width: 2.6rem; }}
.info {{ flex: 1; min-width: 0; }}
.marca {{ font-size: .7rem; text-transform: uppercase; letter-spacing: .9px; color: #A8958B; font-weight: 600; }}
.nombre {{ font-weight: 700; font-size: 1rem; line-height: 1.25; margin: .15rem 0 .35rem 0; color: {CREMA}; }}
.cat {{ font-size: .78rem; color: #A8958B; margin-bottom: .55rem; }}
.fila {{ display: flex; gap: .45rem; flex-wrap: wrap; align-items: center; }}
.precio {{ background: {ROSA}; color: {NEGRO}; border-radius: 999px; padding: .2rem .7rem; font-weight: 700; font-size: .8rem; }}
.dato {{ font-size: .78rem; color: #DCCFC7; }}
.chip {{ border: 1px solid #4A3A33; border-radius: 999px; padding: .12rem .55rem; font-size: .68rem; color: #DCCFC7; }}
.chips {{ display: flex; gap: .35rem; flex-wrap: wrap; margin-top: .6rem; }}

/* pie */
.pie h3 {{ font-family: 'Anton', sans-serif; font-weight: 400; text-transform: uppercase; color: {ROSA}; font-size: 2rem; margin: 2.4rem 0 .6rem 0; }}
.pie-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }}
.pie-grid div {{ border: 1.5px solid #3A2C26; border-radius: 22px; padding: 1rem 1.2rem; font-size: .86rem; color: #DCCFC7; }}
.pie-grid b {{ display: block; font-family: 'Anton', sans-serif; font-weight: 400; color: {CREMA}; font-size: 1.6rem; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Cargando la red y el catálogo (unos segundos)...")
def cargar():
    s = load_splits()
    idx = ItemIndex(s.items)
    todo = pd.concat([s.train, s.val_warm, s.val_cold], ignore_index=True)
    tt = TwoTowerRecommender(todo, s.users, s.items, idx, weights=ROOT / "models" / "two_tower_final.pt",
                             dim=128, hidden=512, threads=4)
    items = s.items.reset_index(drop=True).copy()
    items["positivos"] = np.bincount(idx.to_pos(todo.product_id), minlength=len(idx))
    items["etiqueta"] = items.product_name + " · " + items.brand_name
    items["highlights_l"] = items.highlights.map(_parse_list)
    items["ranking_pop"] = items.positivos.rank(ascending=False, method="min").astype(int)
    return tt, items


def opciones(d):
    return [NO_SE] + list(d)


def nombre(d):
    return lambda k: "prefiero no decir" if k == NO_SE else d[k]


e = html.escape


def html_(s):
    """Compacta el HTML en una línea: evita que el Markdown de Streamlit lo tome como bloque de código."""
    return " ".join(line.strip() for line in s.splitlines() if line.strip())


def mostrar(s):
    st.markdown(html_(s), unsafe_allow_html=True)
tt, items = cargar()

# ---------------------------------------------------------------- barra lateral: tu piel
with st.sidebar:
    st.header("Tu piel")
    tono = st.selectbox("Tono", opciones(TONOS), index=4, format_func=nombre(TONOS))
    tipo = st.selectbox("Tipo de piel", opciones(TIPOS), index=3, format_func=nombre(TIPOS))
    ojos = st.selectbox("Color de ojos", opciones(OJOS), index=1, format_func=nombre(OJOS))
    pelo = st.selectbox("Color de pelo", opciones(PELO), index=1, format_func=nombre(PELO))

    st.header("Lo que ya probaste")
    elegidos = st.multiselect("Productos que te gustaron (opcional)", items.index,
                              format_func=lambda i: items.etiqueta[i], placeholder="Buscá por nombre o marca")

    st.header("Preferencias")
    variedad = st.select_slider("Variedad", options=list(VARIEDAD), value=0.25, format_func=VARIEDAD.get,
                                help="Penaliza los productos más populares para mostrar opciones menos conocidas. "
                                     "Más variedad = un poco menos de acierto (ver notebook 04).")
    categorias = st.multiselect("Categorías", sorted(items.secondary_category.dropna().unique()),
                                placeholder="Todas")
    precio_max = st.slider("Precio máximo (USD)", 5, int(items.price_usd.max()), 150, step=5)

# ---------------------------------------------------------------- recomendar
perfil = pd.DataFrame([{"skin_tone": tono, "skin_type": tipo, "eye_color": ojos, "hair_color": pelo}])
logits = tt.score_profiles(perfil, [list(elegidos)])[0] / 0.05 - variedad * np.log1p(items.positivos.to_numpy())

ok = (items.price_usd <= precio_max).to_numpy().copy()
if categorias:
    ok &= items.secondary_category.isin(categorias).to_numpy()
ok[list(elegidos)] = False
logits = np.where(ok, logits, -np.inf)
top = [i for i in np.argsort(-logits)[:10] if np.isfinite(logits[i])]

# ---------------------------------------------------------------- encabezado
modo = "tu perfil + historial" if elegidos else "solo tu perfil"
piel = " · ".join(x for x in [TIPOS.get(tipo), TONOS.get(tono, "").split(" · ")[0]] if x) or "sin datos"
mostrar(f"""
<div class="meta">
  <div>Modelo<b>Two-tower · PyTorch</b></div>
  <div>Datos<b>~1M reseñas de Sephora</b></div>
  <div>Recomendando con<b>{e(modo)}</b></div>
  <div>Tu piel<b>{e(piel)}</b></div>
</div>
<div class="hero">
  <h1>Tu top 10<br>de skincare</h1>
  <div class="pills">
    <span class="pill blanca">{'con historial' if elegidos else 'sin historial'}</span>
    <span class="pill rosa">variedad: {e(VARIEDAD[variedad])}</span>
    <span class="pill borde">{len(top)} productos</span>
  </div>
</div>
<p class="sub">Una red neuronal con dos torres, una para vos y otra para cada producto, aprendió de un millón de
reseñas qué le gusta a cada tipo de piel. Contale de la tuya en la barra rosa.</p>
""")

if tono in TONOS_SESGADOS:
    mostrar(f"""<div class="aviso"><b>Ojo con el tono {e(tono)}</b>
    En los datos, las pocas personas con este tono reseñan casi solo productos de dos marcas (Caudalie y KORRES).
    La red aprendió ese patrón, que probablemente viene de cómo se juntaron las reseñas y no de lo que necesita
    tu piel (notebook 04, sección 2).</div>""")
if tono == NO_SE and tipo == NO_SE and not elegidos:
    mostrar("""<div class="nota">Sin datos de tu piel ni productos, la red recomienda lo que le gusta a casi
    todo el mundo. Contale algo de vos en la barra rosa.</div>""")
if not top:
    mostrar("""<div class="aviso"><b>Nada por acá</b>Ningún producto cumple los filtros. Probá con otra
    categoría o un precio máximo más alto.</div>""")

# ---------------------------------------------------------------- tarjetas
tarjetas = []
for pos, i in enumerate(top, 1):
    p = items.loc[i]
    cat = e(p.secondary_category) + (f" › {e(p.tertiary_category)}" if isinstance(p.tertiary_category, str) else "")
    chips = "".join(f'<span class="chip">{e(h)}</span>' for h in [h for h in p.highlights_l if not h.startswith("allure")][:4])
    tarjetas.append(f"""
<div class="card{' top' if pos == 1 else ''}">
  <div class="rank">{pos:02d}</div>
  <div class="info">
    <div class="marca">{e(p.brand_name)}</div>
    <div class="nombre">{e(p.product_name)}</div>
    <div class="cat">{cat}</div>
    <div class="fila"><span class="precio">USD {p.price_usd:,.0f}</span>
      <span class="dato">★ {p.rating:.1f}</span><span class="dato">· #{p.ranking_pop} en popularidad</span></div>
    {f'<div class="chips">{chips}</div>' if chips else ''}
  </div>
</div>""")
mostrar(f'<div class="grid">{"".join(tarjetas)}</div>')

# ---------------------------------------------------------------- cómo funciona
mostrar("""
<div class="pie">
<h3>¿Cómo funciona?</h3>
<div class="pie-grid">
  <div><b>Dos torres</b>Una convierte tu perfil (y lo que elegiste) en un vector de 128 números; la otra convierte
  cada producto (marca, categoría, precio, ingredientes, highlights) en otro. Te recomienda los más alineados.</div>
  <div><b>10,2%</b>Sin historial, en datos que nunca vio, la red mete en el top 10 el 10,2% de los productos que le
  gustan a una persona nueva. El mejor modelo simple: 8,9%.</div>
  <div><b>Variedad</b>Resta puntos a los productos más reseñados. "Un poco" casi no cuesta acierto; "mucha" cuesta
  bastante.</div>
</div>
</div>
""")
