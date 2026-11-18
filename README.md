# Recomendador de Skincare

**Pregunta:** Si me decís tu tipo y tono de piel, ¿te puedo recomendar skincare que te vaya a gustar,
aunque nunca hayas reseñado nada? ¿Y funciona igual de bien para todos los tonos de piel?

Modelo principal: una red **two-tower** (persona ↔ producto) en PyTorch, comparada contra baselines clásicos,
con un análisis de *fairness* por tono de piel.

## Datos

Kaggle, "Sephora Products and Skincare Reviews" (nadyinky), en `data/raw/sephora/`:
- `product_info.csv`: 8.494 productos (precio, marca, categoría, ingredientes, highlights)
- `reviews_*.csv`: 1.094.411 reseñas de skincare, 503.216 personas, 2.351 productos reseñados

Primera mirada (17/11/2026):
- 58% de las personas tiene **una sola** reseña; 21% tiene ≥3; 8% tiene ≥5 → el *cold start* es la norma
- Faltan datos de tono de piel en el 15,6% de las reseñas y de tipo de piel en el 10,2%
- Tonos muy desbalanceados: light/fair/lightMedium dominan; deep ~20 mil, rich ~5 mil, dark ~500 → la pregunta de fairness es relevante
- 82% de los ratings son ≥4 → hay que definir con cuidado qué es "le gustó"

## Estructura

```
data/raw/sephora/   datos originales (no se editan)
data/interim/       limpiezas intermedias
data/processed/     interacciones y features listas para modelar
notebooks/          00_, 01_, ... en orden
src/                código reutilizable (datos, features, modelos, evaluación)
models/             pesos entrenados
reports/figures/    gráficos
web/                demo (Streamlit/Gradio) o nota final
docs/               notas y otras ideas (ej. idea_dupes_labiales.md)
```

## Plan

### Paso 0 · ¿Los datos aguantan el proyecto? (semana 1)
- [ ] `00_revision_datos.ipynb`: reseñas por persona y por producto, faltantes, tonos, ratings, fechas
- [ ] Decidir el umbral de actividad para el set "con historial" (¿≥3 reseñas?)

### Paso 1 · Definir el problema (semana 1)
- [ ] Positivo = reseñó el producto **y** le gustó (rating ≥4 o `is_recommended`). Negativos: muestreados
- [ ] Split temporal: la última reseña de cada persona va a test (leave-last-out)
- [ ] Split cold start: personas nunca vistas, solo con su perfil de piel
- [ ] Métricas: Recall@10, NDCG@10, total y **por tono de piel**; cobertura de catálogo

### Paso 2 · Baselines (semana 1)
- [ ] Popularidad global
- [ ] Popularidad por tipo × tono de piel
- [ ] Factorización de matrices (`implicit`)
- [ ] LightFM híbrido (con features)

### Paso 3 · Two-tower (semana 2)
- [ ] Torre persona: embeddings de tono, tipo de piel, ojos, pelo + promedio de los productos del historial
- [ ] Torre producto: embeddings de ID, marca, categoría + log(precio) + ingredientes (Ingredient2Vec)
- [ ] Pérdida softmax con negativos dentro del batch

### Paso 4 · Experimentos (semana 3)
- [ ] Ablaciones: sin perfil, sin ingredientes, sin historial
- [ ] Fairness: recall por tono de piel, ¿el perfil achica la brecha?
- [ ] Sesgo de popularidad
- [ ] Demo: elegís tu piel y te devuelve el top 10

### Paso 5 · Contarlo (semana 3)
- [ ] Modelo vs. baselines · brecha por tono · UMAP de productos

## Entorno

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```
