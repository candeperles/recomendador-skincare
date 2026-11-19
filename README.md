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

### Paso 0 · ¿Los datos aguantan el proyecto? (semana 1) ✅
- [x] `00_revision_datos.ipynb`: reseñas por persona y por producto, faltantes, tonos, ratings, fechas
- [x] Umbral del set "con historial": ≥3 positivos (83 mil personas)
- Decisiones completas al final del notebook 00. Resumen: positivo = rating ≥4; 5 grupos de tono
  (claro / light / medio / tan / oscuro) porque porcelain, olive, dark y ebony son muy chicos;
  desempate al azar cuando la última fecha tiene varias reseñas (35% de los casos)

### Paso 1 · Definir el problema (semana 1) ✅
- [x] Positivo = rating ≥4
- [x] Split con historial por días: test = 1 producto del último día, val = 1 del anteúltimo, train = el resto
  (personas con positivos en ≥3 días: 41.509). Los otros productos de esos días se descartan (3,1%)
- [x] Split cold start: 10% de las personas fuera de train, mitad val y mitad test (~21.500 cada uno)
- [x] 7 chequeos automáticos (sin fugas, orden temporal, reproducible). Todo en `src/splits.py`
  y `notebooks/01_splits.ipynb`; se guarda en `data/processed/`

### Paso 2 · Baselines (semana 1) ✅
- [x] Evaluación común en `src/evaluate.py`: Recall@10, NDCG@10, cobertura, desglose por grupo de tono
- [x] Popularidad global · popularidad por segmento · ALS (`implicit`) · híbrido perfil → ALS
  (reemplaza a LightFM, que no se instala en Windows). Código en `src/baselines.py`
- [x] Resultados en validación (`notebooks/02_baselines.ipynb`, `reports/02_baselines_val*.csv`):
  - Con historial: **ALS 16,9%** de recall@10 (popularidad: 4,7%)
  - Cold start, perfil completo: **pop. por segmento 9,2% ≈ híbrido 9,3%** (popularidad: 7,8%)
  - Sin brecha en contra de los tonos oscuros en los baselines
- Hallazgo: los perfiles incompletos son de reseñas viejas (2011–2016) y más fáciles de adivinar.
  La comparación principal en cold start se hace sobre personas con perfil completo

### Paso 3 · Two-tower (semana 2) ✅
- [x] Torre persona: embeddings de tono, tipo de piel, ojos, pelo + promedio de los vectores de los productos del historial
- [x] Torre producto: embeddings de ID, marca, categorías + log(precio) + ingredientes y highlights
  (aprendidos dentro de la red, en lugar de un Ingredient2Vec aparte)
- [x] Pérdida softmax sobre **todo** el catálogo (con 2.351 productos no hace falta muestrear negativos)
- [x] Código en `src/two_tower.py`, grilla en `scripts/grid_two_tower.py`, análisis en `notebooks/03_two_tower.ipynb`
- Resultados en validación (`reports/03_two_tower_val.csv`):
  - Con historial: **18,5%** de recall@10 (ALS: 16,9%; +1,6 ± 0,3 puntos)
  - Cold start, perfil completo: **10,3%** (híbrido: 9,3%; +1,0 ± 0,3 puntos)
  - Sin brecha en contra de los tonos oscuros: el grupo *oscuro* es el que más mejora
  - Punto débil: en cold start recomienda pocos productos distintos (67), casi todos populares

### Paso 4 · Experimentos (semana 3)
- [ ] Ablaciones: sin perfil, sin ingredientes/highlights, sin historial, sin ID
- [ ] Fairness: recall por tono de piel; ¿por qué *dark* recibe una lista tan distinta y *tan* no mejora?
- [ ] Sesgo de popularidad: penalizar lo popular en cold start, ¿cuánto recall cuesta?
- [ ] Evaluación final en test (una sola vez)
- [ ] Demo: elegís tu piel y te devuelve el top 10

### Paso 5 · Contarlo (semana 3)
- [ ] Modelo vs. baselines · brecha por tono · UMAP de productos

## Entorno

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```
