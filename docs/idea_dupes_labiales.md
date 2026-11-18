# Proyecto Dupes Maquillaje

**Pregunta:** ¿Los dupes que recomienda la gente son realmente dupes?
Objetivo: medir de forma objetiva la similitud entre productos (color, acabado, ingredientes, precio),
aprender una métrica de "dupe" a partir de pares etiquetados y contarlo como una nota visual tipo *The Pudding*.

## Estructura

```
data/
  raw/         datos tal como se descargan (no se editan nunca)
  interim/     datos limpios intermedios
  processed/   tabla final de productos/tonos lista para modelar
  labels/      pares dupe / no dupe etiquetados a mano
notebooks/     exploración y prototipos (01_, 02_, ... en orden)
src/           código reutilizable (limpieza, color, features, modelos)
reports/figures/  gráficos
web/           la nota interactiva final
```

## Categoría

Labiales como categoría principal (más tonos y más contenido de dupes).
Con la Makeup API, blush tiene solo ~300 tonos y labial ~1.700.
Blush queda como extensión si sobra tiempo.

## Fases

### Fase 0: Datos (semanas 1-2)
- [x] Makeup API: `data/raw/makeupapi_lipstick.json` y `makeupapi_blush.json` (ya descargados)
- [ ] Kaggle "Sephora Products and Skincare Reviews" (nadyinky) -> `data/raw/sephora/`
- [ ] Open Beauty Facts (ingredientes): solo si hacen falta para los productos elegidos
- [ ] Decidir si hace falta scrapear los colores de los swatches de Sephora (revisar términos de uso)

### Fase 1: Tabla de tonos (semana 2-3)
- [ ] Una fila por tono: marca, producto, nombre del tono, hex, precio, acabado, fuente
- [ ] Hex -> CIELAB (`colormath` o `skimage.color`), distancia ΔE2000
- [ ] Acabado desde el texto (matte / satin / gloss / shimmer / metallic) con reglas de palabras clave

### Fase 2: Etiquetado (semanas 3-5): el aporte original
- [ ] Juntar pares de dupes de Reddit (r/drugstoremua, r/MakeupAddiction), artículos y Temptalia (solo para validar)
- [ ] Formato: `data/labels/pairs.csv` con `producto_a, tono_a, producto_b, tono_b, etiqueta (dupe/parecido/no), fuente, url`
- [ ] Meta: 300-500 pares. Agregar negativos "difíciles" (mismo color, distinto acabado)

### Fase 3: Modelado (semanas 5-7)
- [ ] Baseline 1: solo ΔE2000 con un umbral
- [ ] Baseline 2: regresión logística sobre las diferencias de features (color, acabado, precio, ingredientes)
- [ ] Opcional: red siamesa / triplet loss si hay suficientes pares
- [ ] Evaluación: recall@k para "dado un producto, ¿aparece su dupe en el top-k?"

### Fase 4: La historia (semanas 7-9)
- [ ] Guion en papel: qué ve el lector en cada paso del scroll
- [ ] Gráfico estrella: los labiales en el espacio de color, con los dupes "de TikTok" conectados
- [ ] Web: Quarto + Plotly, o D3 + Scrollama
- [ ] Publicar en GitHub Pages

## Primeros pasos concretos
1. Crear el entorno: `python -m venv .venv` y `pip install pandas numpy scikit-learn scikit-image matplotlib plotly jupyter kaggle`
2. Bajar el dataset de Sephora de Kaggle a `data/raw/sephora/`
3. Notebook `01_explorar_fuentes.ipynb`: ¿cuántos labiales hay en cada fuente? ¿Cuántos tienen hex? ¿Qué marcas se cruzan entre fuentes?
4. Empezar a anotar dupes en `data/labels/pairs.csv` desde ya (es lo más lento)

## Notas sobre las fuentes
- La Makeup API tiene datos de ~2017, sobre todo marcas de farmacia de EE.UU. Sirve para el prototipo, no para productos actuales.
- Respetar robots.txt y términos de uso al scrapear. No redistribuir imágenes ni textos de terceros en la web final.
