# Curso IA Aplicada a Ciencia de Materiales — Notebooks

Construidos y probados el 2026-08-08 (entorno local `~/.venvs/ia-materiales`,
Python 3.12, versiones de `environment.yml`).

## Kabré OnDemand (entorno oficial del curso)

Actualizado el 2026-09-22. Probado en Kabré con el kernel **Python-AI-Materials**
(Python 3.10, numpy 1.26, pymatgen 2024.10, matminer 0.9.3, botorch 0.16,
mattergen 1.0.3): los 13 notebooks corren en kura (sin GPU) y MatterGen corre
en nukwa (GPU).

- **Datasets de matminer:** vienen en `datos/matminer_cache/`. Los notebooks
  fijan `MATMINER_DATA` a esa carpeta (la instalación compartida no es escribible).
- **Visualización:** las extensiones de plotly y nglview están en el ambiente
  python-AI-materials, pero el servidor Jupyter de OnDemand no las ve. La
  primera celda de cada notebook llama a `activar_extensiones()`, que las enlaza
  en `~/.local/share/jupyter/labextensions` del estudiante (la primera vez pide
  recargar con F5) y corrige la versión que pide nglview 4.0.1 (su JavaScript
  empaquetado es 3.1.5). Probado en kura el 26 set 2026. Si algo falla,
  `USAR_PLOTLY = False` y `USAR_NGLVIEW = False` vuelven a matplotlib y ASE.
- **Tareas con GPU (Día 3):** con unos 50 estudiantes, en clase se trabaja en
  kura con estructuras y modelos ya preparados, y la GPU se usa como tarea: cada
  estudiante envía `tareas/tarea_mattergen.sbatch` (generación) y
  `tareas/tarea_vision.sbatch` (entrenamiento corto de la CNN de XRD) a la cola
  nukwa-l40s y revisa el resultado después. Ver `tareas/LEEME.md`.
- **MatterGen (Día 3):** sesión en la cola nukwa. Necesita
  (1) `mattergen_src/` (código fuente v1.0.3, incluido aquí, porque el paquete
  instalado no trae `sampling_conf/` ni `gemnet-dT.json`) y
  (2) los pesos en `/work/$USER/hf_cache` (4 modelos, 1.9 GB) o en la ruta de
  la variable `IA_CM_HF_HOME`. Sin GPU o sin pesos el notebook usa
  `datos/mattergen_fallback/` (en Kabré: CIFs reales generados por MatterGen).
- **Novedad en D3:** `pre_cache_mp.py` también guarda
  `datos/mp_formulas_todas.json` (todas las fórmulas de MP). Sin ese archivo la
  comparación de novedad es aproximada.
- **Visión computacional (Día 3, P3.5):** notebook `06_vision_computacional` con
  dos modelos ya entrenados en Kabré (CNN 1D para XRD y ResNet-18 para SEM).
  Datos y pesos en `datos/vision/` (33 MB, solo en la copia de Kabré:
  `~/ia_cm_prueba/ia_cm_curso_v2_kabre.zip`); scripts de entrenamiento en
  `entrenamiento_vision/`. Corre en CPU.
- **Explicaciones de código:** las celdas "🔍 Cómo funciona este código"
  explican las partes menos obvias (subprocess y variables de entorno,
  tensores de botorch, GroupKFold, SHAP, SMACT, etc.).

## Estructura

> **¿Vas a correr esto en otra máquina o preparar una demo?**
> Ver `INSTRUCCIONES_VSCODE.md` (setup paso a paso, modo offline, guion de demo)
> y `SETUP_DEMO.md`. Usar `requirements_probado.txt` (versiones exactas
> verificadas), no `environment.yml`, hasta que este último se valide.

```
curso/
├── environment.yml            # entorno oficial del curso (conda)
├── requirements_probado.txt   # versiones exactas del entorno de pruebas (pip)
├── INSTRUCCIONES_VSCODE.md    # correr los notebooks en VS Code / otra máquina
├── SETUP_DEMO.md              # setup rápido para demos (esta máquina u otra)
├── pre_cache_mp.py            # generar caché de Materials Project (necesita API key)
├── datos/
│   ├── superhard_experimental.csv     # D4 — 200 exp. PVD sintéticos
│   ├── brief_A_termoelectricos.csv    # D5 — 800 materiales (ZT regenerado 2026-08-08,
│   │                                  #      ahora físicamente consistente con S²σT/κ)
│   ├── brief_B_recubrimiento.csv      # D5 — 35 exp. PVD
│   ├── mp_bandgap_cache.MOCK.json     # ⚠ MOCK para pruebas offline — el real
│   │                                  #   se genera con pre_cache_mp.py.
│   │                                  #   Renombrar a mp_bandgap_cache.json solo
│   │                                  #   para probar sin API key.
│   └── mattergen_fallback/            # ⚠ CIFs PLACEHOLDER (prototipos perturbados).
│                                      #   Reemplazar con salidas reales de MatterGen
│                                      #   del servidor GPU (misma estructura).
├── notebooks/
│   ├── 00_test_entorno.ipynb          # pre-curso (alumnos)
│   ├── 01_exploracion_datos.ipynb     # Día 1
│   ├── 02_sklearn_matminer.ipynb      # Día 2 (requiere pickle del D1 + caché MP)
│   ├── 03_mattergen_generacion.ipynb  # Día 3 (GPU; fallback automático sin GPU)
│   ├── 04_doe_bayesiana.ipynb         # Día 4
│   ├── 05_brief_A_termoelectrico.ipynb  # Día 5
│   ├── 05_brief_B_recubrimiento.ipynb   # Día 5
│   ├── soluciones/                    # instructor (ejercicios D1-D4 + briefs)
│   └── _ejecutados/                   # copias ejecutadas (verificación, con outputs)
├── tareas/                            # tareas con GPU (sbatch a nukwa): MatterGen y CNN de XRD
└── src/                               # fuentes jupytext (py:percent) de todo
```

## Flujo de dependencias entre días

- D1 guarda `datos/df_con_features_dia1.pkl` (~8 MB) → D2 lo carga.
- D2 necesita `datos/mp_bandgap_cache.json` → generarlo con `pre_cache_mp.py`
  antes del curso (o renombrar el MOCK para pruebas).
- D3 usa MatterGen en GPU (nukwa); sin GPU carga `datos/mattergen_fallback/` automáticamente.
- D3 y sol_dia3 comparten `salidas_mattergen/`: la solución reutiliza los lotes que generó el notebook 03.
- Soluciones D2/D3 también usan el caché MP.

## Tiempos de ejecución medidos (CPU laptop, sin tiempo humano)

| Notebook | Ejecución completa | Bloque de clase |
|----------|-------------------|-----------------|
| 00_test_entorno | 14 s | pre-curso |
| 01 (D1) | 13 s | 2 h |
| 02 (D2) | 27 s | 2 h |
| 03 (D3, fallback) | 6 s | 2 h (GPU: generación ~min por lote) |
| 04 (D4, 3 semillas en Hartmann) | 75 s | 2 h |
| 05 A/B | <10 s (esqueleto) | 2 h |
| sol_dia4 (2 campañas BO + retos) | ~1.5 min | — |

El cómputo nunca es el cuello de botella: los bloques de 2 h están dominados
por explicación, ejercicios y discusión, como debe ser.

## Mantenimiento

Los notebooks se editan en `src/*.py` (formato jupytext py:percent) y se
convierten con:

```bash
jupytext --to ipynb src/XX.py -o notebooks/XX.ipynb
```
