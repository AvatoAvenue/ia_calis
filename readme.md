# Instalación
- dependencias

```
pip install torch pandas g2p-en nltk
python -m nltk.downloader averaged_perceptron_tagger_eng cmudict
```

# Flujo de trabajo

## Agregar datos de entrenamiento

Edita `dataset.csv` directamente o usa el CLI interactivo:

```bash
python g2p_cli.py
# Agregar palabras al dataset
# Generar IPA automático con g2p_en
```

**Formato del dataset:**

```
text,ipa
server,ˈsɜrvər
database,ˈdeɪtəˌbeɪs
```

- Las entradas con `*` en el IPA (p.ej. `docker,docker*`) son ignoradas durante el entrenamiento. Úsalas para acrónimos o términos sin IPA estándar.

## Entrenar el modelo

```bash
python entrenar.py
```

**Opciones disponibles:**

```bash
python entrenar.py --epochs 800 --lr 0.0005 --batch 16
python entrenar.py --dataset dataset.csv --model g2p_model.pt
```

| Argumento  | Default       | Descripción                      |
|------------|---------------|----------------------------------|
| --dataset  | dataset.csv   | CSV de entrenamiento             |
| --model    | g2p_model.pt  | Ruta donde guardar el modelo     |
| --vocab    | g2p_vocab.json| Ruta donde guardar el vocabulario|
| --epochs   | 500           | Número de épocas                 |
| --lr       | 0.001         | Learning rate                    |
| --batch    | 32            | Tamaño de batch                  |
| --embed    | 64            | Dimensión de embeddings          |
| --hidden   | 256           | Dimensión de capas ocultas LSTM  |

---

## Usar el modelo — CLI interactivo

```bash
python g2p_cli.py
```

**Menú disponible:**

| Opción | Función |
|--------|---------|
| **1** | Predecir IPA de cualquier palabra |
| **2** | Comparar predicción del modelo vs referencia del dataset + g2p_en |
| **3** | Agregar palabras manualmente con IPA a mano |
| **4** | Ver todas las palabras del dataset (paginado) |
| **5** | Buscar una palabra en el dataset |
| **6** | Generar IPA automático para múltiples palabras usando g2p_en |
| **7** | Estadísticas del dataset y estado del sistema |
| **0** | Salir |

### 4. Evaluar el modelo

```bash
python evaluar.py
python evaluar.py --top 20 --export
```

Genera un reporte con:
- Precisión exacta
- Similitud media
- Tabla de los mejores y peores casos
- Exportación opcional a `evaluacion_resultado.csv`

---

## Ciclo de mejora continua

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   dataset.csv  ──►  entrenar.py  ──►  g2p_model.pt  │
│       ▲                                     │       │
│       │                                     ▼       │
│   g2p_cli.py  ◄──────────────────  evaluar.py       │
│   (agregar)                        (detectar         │
│                                     errores)         │
└─────────────────────────────────────────────────────┘
```

1. **Evalúa** con `evaluar.py` para encontrar palabras mal predichas
2. **Agrega** más ejemplos similares con `g2p_cli.py` → Opción 3 o 6
3. **Reentrena** con `entrenar.py`
4. **Repite**
