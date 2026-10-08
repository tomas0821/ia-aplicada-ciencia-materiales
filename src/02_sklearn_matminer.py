# %% [markdown]
# <div style="background:linear-gradient(90deg,#1A2E4A 0%,#2E5E9B 100%);color:white;padding:20px 26px;border-radius:12px;">
# <span style="font-size:12px;letter-spacing:3px;color:#D6E8F7;">CURSO IA APLICADA A CIENCIA DE MATERIALES &nbsp;·&nbsp; DÍA 2 · PREDICCIÓN</span><br>
# <span style="font-size:26px;font-weight:700;">🎯 Predicción de Propiedades y Diseño Inverso</span><br>
# <span style="font-size:14px;color:#D6E8F7;">De horas de DFT a milisegundos de ML — y de la propiedad al material</span>
# </div>
#
# | Bloque | Contenido | Tiempo |
# |--------|-----------|--------|
# | P2.1 | Pipeline ML completo (Random Forest) | 35 min |
# | P2.2 | API de Materials Project | 30 min |
# | P2.3 | Diseño inverso básico | 35 min |
# | P2.4 | SHAP: interpretabilidad | 15 min |
#
# **La idea central de hoy:** ayer convertimos materiales en números. Hoy
# entrenamos un modelo que **predice el módulo de compresibilidad en
# milisegundos** (un cálculo DFT tarda horas) — y luego invertimos la pregunta:
# en vez de "¿qué propiedades tiene este material?", preguntamos
# **"¿qué material tiene las propiedades que quiero?"**

# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>P2.1 — Pipeline ML completo</b> <span style="float:right;background:#E87C1E;color:white;padding:2px 12px;border-radius:12px;font-size:12px;font-weight:600;">⏱ 35 min</span></div>
#
# El flujo estándar de *cualquier* proyecto de ML supervisado:
#
# **datos → split train/test → escalar → entrenar → evaluar → interpretar**
#
# Cargamos las features Magpie que calculamos ayer:


# %%
# --- 🎨 Estilo visual del curso (paleta validada para accesibilidad) ---
import matplotlib.pyplot as plt
from cycler import cycler

AZUL, NARANJA, PURPURA, VERDE = "#2E5E9B", "#D96F0F", "#B0509E", "#2F9E44"
TINTA, GRIS = "#1A2E4A", "#8A94A3"
plt.rcParams.update({
    "axes.prop_cycle": cycler(color=[AZUL, NARANJA, PURPURA, VERDE]),  # orden fijo
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": TINTA, "axes.labelcolor": TINTA, "text.color": TINTA,
    "xtick.color": TINTA, "ytick.color": TINTA,
    "axes.titleweight": "bold", "axes.titlesize": 13, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
    "lines.linewidth": 2.0, "figure.dpi": 105, "legend.frameon": False,
})

# --- Opciones de visualización (Kabré OnDemand) ---
# plotly (gráficos interactivos) y nglview (visor 3D) necesitan que el servidor
# Jupyter vea sus extensiones. activar_extensiones() las enlaza en su carpeta
# personal (~/.local) y ajusta la versión del visor 3D. Si imprime un aviso,
# recargue la página (F5) una sola vez y siga. Si algo falla, ponga ambas
# banderas en False: se usan matplotlib y ASE.
USAR_PLOTLY = True
USAR_NGLVIEW = True


def activar_extensiones():
    import json, sys
    from pathlib import Path
    fuente = Path(sys.prefix) / "share" / "jupyter" / "labextensions"
    destino = Path.home() / ".local" / "share" / "jupyter" / "labextensions"
    nuevas = []
    for ext in ["jupyterlab-plotly", "nglview-js-widgets"]:
        if (fuente / ext).is_dir() and not (destino / ext).exists():
            destino.mkdir(parents=True, exist_ok=True)
            (destino / ext).symlink_to(fuente / ext)
            nuevas.append(ext)
    if nuevas:
        print("Extensiones activadas:", ", ".join(nuevas))
        print("👉 Recargue la página (F5) una vez y continúe.")
    paquete = fuente / "nglview-js-widgets" / "package.json"
    if not (USAR_NGLVIEW and paquete.is_file()):
        return
    # nglview 4.0.1 pide la parte JavaScript "4.0", pero el paquete instalado
    # trae la 3.1.5 (error de empaquetado). Pedimos la versión que sí existe.
    import ipywidgets
    with ipywidgets.Output():
        import nglview  # noqa: F401  (al importarse muestra un widget; lo ocultamos)
    version_js = json.loads(paquete.read_text())["version"]

    def subclases(c):
        for s in c.__subclasses__():
            yield s
            yield from subclases(s)

    for c in set(subclases(ipywidgets.Widget)):
        t = c.class_traits()
        if "_model_module" in t and t["_model_module"].default_value == "nglview-js-widgets":
            for n in ("_model_module_version", "_view_module_version"):
                if n in t:
                    t[n].default_value = version_js
                    iniciales = getattr(c, "_static_immutable_initial_values", None)
                    if iniciales is not None and n in iniciales:
                        iniciales[n] = version_js


activar_extensiones()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

def encontrar_datos() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents[:3]]:
        if (base / "datos").is_dir():
            return base / "datos"
    raise FileNotFoundError("No encuentro la carpeta datos/")

DATOS = encontrar_datos()

df = pd.read_pickle(DATOS / "df_con_features_dia1.pkl")
print(f"Dataset del Día 1: {df.shape}")
# ¿Perdiste el archivo? Ejecuta de nuevo la sección P1.4 del notebook del Día 1.

# %% [markdown]
# **Features (X) y target (y).** X = las 132 columnas Magpie; y = K_VRH.
# Quitamos las filas donde la featurización falló (`dropna`).

# %%
feature_cols = [c for c in df.columns if "MagpieData" in c]
X = df[feature_cols].dropna()
y = df.loc[X.index, "K_VRH"]

print(f"Shape X: {X.shape}, y: {y.shape}")
print(f"Distribución de y: media={y.mean():.1f} GPa, std={y.std():.1f} GPa")

# %% [markdown]
# **Split train/test.** La regla de oro del ML: el modelo se evalúa SOLO con
# datos que **nunca vio durante el entrenamiento**. 80% entrenar / 20% evaluar.
#
# **Escalado.** Cada feature a media 0 y varianza 1. (Random Forest no lo
# necesita estrictamente, pero es el hábito correcto — los GP del Día 4 sí lo
# exigen.) Fíjense: el scaler se ajusta con el train y se *aplica* al test —
# nunca al revés, eso sería filtrar información del test al modelo.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"Train: {X_train_sc.shape[0]} materiales | Test: {X_test_sc.shape[0]} materiales")

# %% [markdown]
# <div style="background:#EEF3F9;border-left:6px solid #2E5E9B;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#2E5E9B;">🔍 Cómo funciona este código</b></div>
#
# - `df.loc[X.index, "K_VRH"]`: después de `dropna()` algunas filas
#   desaparecen; usar el mismo índice mantiene alineados X e y.
# - `fit_transform` en train **aprende** medias y desviaciones y las aplica;
#   `transform` en test **solo aplica** las del train. Si el escalador viera el
#   test estaríamos filtrando información y el error parecería menor.
# - `random_state=42` fija el azar: todos obtienen el mismo resultado.
# - **Polimorfos:** 174 materiales comparten fórmula con otro y reciben el
#   mismo vector Magpie. Un split aleatorio puede poner uno en train y su
#   "gemelo" en test. `GroupKFold` con la fórmula como grupo evita esa fuga
#   (lo verán en el Reto 1).

# %% [markdown]
# **Entrenar el Random Forest** — 200 árboles de decisión que votan.
# En una laptop moderna esto tarda segundos:

# %%
rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
rf.fit(X_train_sc, y_train)
y_pred = rf.predict(X_test_sc)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.1f} GPa  |  R²: {r2:.3f}")
# Esperado: MAE ≈ 10-14 GPa, R² ≈ 0.88-0.92

# %% [markdown]
# ### Parity plot — la figura #1 del ML de materiales
#
# Predicción vs valor real. Un modelo perfecto pondría todos los puntos sobre
# la diagonal. Miren *dónde* se desvía: ¿en valores altos? ¿bajos?
#
# Con `USAR_PLOTLY = True` la figura es interactiva (el mouse muestra el
# material). Los peores errores se ven con
# `df.loc[y_test.index].assign(err=abs(y_test - y_pred)).nlargest(5, "err")`.

# %%
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
try:
    if not USAR_PLOTLY:
        raise ImportError("plotly desactivado en Kabré")
    import plotly.express as px

    df_parity = pd.DataFrame({
        "K_DFT": y_test.values, "K_ML": y_pred,
        "formula": df.loc[y_test.index, "formula"].values,
        "error_GPa": np.abs(y_test.values - y_pred).round(1),
    })
    figx = px.scatter(
        df_parity, x="K_DFT", y="K_ML", hover_name="formula",
        hover_data={"K_DFT": ":.0f", "K_ML": ":.0f", "error_GPa": True},
        opacity=0.6, color_discrete_sequence=[AZUL],
        labels={"K_DFT": "K_VRH DFT (GPa)", "K_ML": "K_VRH predicho (GPa)"},
        title=f"Parity plot — MAE={mae:.1f} GPa, R²={r2:.3f}",
    )
    figx.add_shape(type="line", x0=lims[0], y0=lims[0], x1=lims[1], y1=lims[1],
                   line=dict(color=GRIS, dash="dash", width=1))
    figx.update_traces(marker=dict(size=7))
    figx.update_layout(width=640, height=620, font_color=TINTA,
                       plot_bgcolor="white", title_font_weight=700)
    figx.update_xaxes(gridcolor="#E8EAEE")
    figx.update_yaxes(gridcolor="#E8EAEE")
    figx.show()
except ImportError:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.5, s=20)
    ax.plot(lims, lims, "--", color=GRIS, lw=1, label="y = x (modelo perfecto)")
    ax.set_xlabel("K_VRH DFT (GPa)")
    ax.set_ylabel("K_VRH predicho por ML (GPa)")
    ax.set_title(f"Parity plot — MAE={mae:.1f} GPa, R²={r2:.3f}")
    ax.legend()
    plt.tight_layout()
    plt.show()

# 🤔 El DFT que generó estos datos tarda ~horas por material. El RF predice
#    1,000 materiales en <1 segundo. ¿Qué pierde uno a cambio?

# %% [markdown]
# ### ¿Qué features usa el modelo?

# %%
importances = pd.Series(rf.feature_importances_, index=feature_cols)
importances.nlargest(15).sort_values().plot(
    kind="barh", figsize=(8, 6), edgecolor="white",
    title="Top 15 features por importancia (RF)",
)
plt.xlabel("Importancia relativa")
plt.tight_layout()
plt.show()

# La temperatura de fusión promedio y los electrones no apareados dominan —
# ambos son proxies de la fuerza del enlace. El modelo "redescubrió" física.

# %% [markdown]
# <div style="background:#FFF3E6;border-left:6px solid #E87C1E;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#C25E08;">🎯 Reto 1 &nbsp;·&nbsp; 10 min</b></div>
#
# **Reto:** ¿el modelo es estable o tuvimos suerte con el split?
#
# 1. Reporta el R² con **validación cruzada de 5 folds** (`cross_val_score`
#    ya está importado — cómo se usa es investigación suya:
#    la documentación de scikit-learn es su amiga de por vida).
# 2. Luego intenta **bajar el MAE por debajo de 14 GPa** ajustando
#    hiperparámetros del RandomForest (referencia: con 500 árboles y
#    `max_features=0.3` se llega a ≈ 13.9 GPa). ¿Cuáles moviste y cuánto
#    ganaste?
#    ¿En qué punto dejó de mejorar?

# %%
# [Su código aquí — parte 1: validación cruzada]

# %%
# [Su código aquí — parte 2: hiperparámetros]


# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>P2.2 — API de Materials Project</b> <span style="float:right;background:#E87C1E;color:white;padding:2px 12px;border-radius:12px;font-size:12px;font-weight:600;">⏱ 30 min</span></div>
#
# [Materials Project](https://next-gen.materialsproject.org) contiene
# propiedades DFT de >150,000 materiales, accesibles por API. En clase usamos
# un **caché local pre-descargado** (`datos/mp_bandgap_cache.json`) para no
# depender de la red — el bloque `else` muestra cómo se descargó.
#
# > 🔑 Para usar la API en casa: crea una cuenta gratuita, copia tu API key a
# > un archivo `.env` (`MP_API_KEY=...`) y ejecuta `pre_cache_mp.py`.

# %%
import os
import json
from dotenv import load_dotenv

load_dotenv()  # carga MP_API_KEY desde .env si existe
API_KEY = os.getenv("MP_API_KEY")

CACHE_FILE = DATOS / "mp_bandgap_cache.json"

if CACHE_FILE.exists():
    df_mp = pd.read_json(CACHE_FILE)
    print(f"✓ Cargado desde caché: {len(df_mp)} materiales")
elif API_KEY:
    from mp_api.client import MPRester

    with MPRester(API_KEY) as mpr:
        docs = mpr.materials.summary.search(
            band_gap=(0.3, 2.5), is_stable=True,
            fields=["material_id", "formula_pretty", "band_gap",
                    "formation_energy_per_atom", "energy_above_hull",
                    "symmetry", "nsites"],
        )
    df_mp = pd.DataFrame({
        "material_id": str(d.material_id),
        "formula": d.formula_pretty,
        "band_gap": d.band_gap,
        "Ef": d.formation_energy_per_atom,
        "ehull": d.energy_above_hull,
        "crystal_system": str(d.symmetry.crystal_system) if d.symmetry else None,
        "spacegroup": d.symmetry.symbol if d.symmetry else None,
        "nsites": d.nsites,
    } for d in docs)
    df_mp.to_json(CACHE_FILE, orient="records")
    print(f"✓ Descargado y cacheado: {len(df_mp)} materiales")
else:
    raise RuntimeError(
        "No hay caché ni API key. En clase: pide el archivo "
        "mp_bandgap_cache.json. En casa: configura .env con MP_API_KEY."
    )

df_mp.head(3)

# %% [markdown]
# **EDA rápido del resultado** — ¿qué nos trajo la consulta?
# (El filtro fue: bandgap entre 0.3 y 2.5 eV + termodinámicamente estable —
# el rango interesante para absorbedores solares y termoeléctricos.)

# %%
print(df_mp["crystal_system"].value_counts())

df_mp["band_gap"].hist(bins=50, figsize=(8, 4), edgecolor="white")
plt.xlabel("Bandgap (eV)")
plt.ylabel("N materiales")
plt.title("Distribución de bandgap (materiales estables, 0.3–2.5 eV)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>P2.3 — Diseño inverso básico</b> <span style="float:right;background:#E87C1E;color:white;padding:2px 12px;border-radius:12px;font-size:12px;font-weight:600;">⏱ 35 min</span></div>
#
# **El giro conceptual del curso.** Hasta ahora: material → propiedad.
# Ahora: propiedad deseada → ¿qué material?
#
# La receta más simple (*screening virtual*):
# 1. **Generar** un pool de composiciones candidatas (aquí: carburos, nitruros,
#    boruros... de metales de transición — la familia de los materiales duros)
# 2. **Featurizar** cada candidato con Magpie
# 3. **Predecir** K_VRH con el modelo de P2.1
# 4. **Filtrar** los mejores y verificar cuáles son *nuevos*

# %%
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition

elements_transition = ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W"]
elements_light = ["C", "N", "B", "O", "Si", "Al"]

candidates = []
for A in elements_transition:
    for B in elements_light:
        for x in [1, 2, 3]:
            for y_ in [1, 2]:
                formula = f"{A}{x}{B}{y_}"
                try:
                    comp = Composition(formula)
                    if comp.valid:
                        candidates.append({
                            "formula": formula,
                            "reduced": comp.reduced_formula,
                            "composition": comp,
                        })
                except Exception:
                    pass

df_cand = pd.DataFrame(candidates)
print(f"Candidatos generados: {len(df_cand)}")

# Ti1N1 y Ti2N2 son el MISMO material — eliminamos duplicados por fórmula reducida
df_cand = df_cand.drop_duplicates("reduced").reset_index(drop=True)
print(f"Candidatos únicos (fórmula reducida): {len(df_cand)}")

# %% [markdown]
# **Featurizar y predecir** — reutilizamos el `scaler` y el `rf` ya entrenados
# (por eso importaba no cerrar el kernel):

# %%
ep = ElementProperty.from_preset("magpie")
import joblib
ep.set_n_jobs(joblib.cpu_count())  # solo los núcleos de esta sesión, no los 40 del nodo
df_cand = ep.featurize_dataframe(df_cand, col_id="composition", ignore_errors=True, pbar=False)

fc = [c for c in df_cand.columns if "MagpieData" in c]
X_cand = df_cand[fc].dropna()
X_cand_sc = scaler.transform(X_cand)  # ¡el MISMO scaler del entrenamiento!

df_cand.loc[X_cand.index, "K_VRH_pred"] = rf.predict(X_cand_sc)

# %% [markdown]
# <div style="background:#EEF3F9;border-left:6px solid #2E5E9B;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#2E5E9B;">🔍 Cómo funciona este código</b></div>
#
# - Los cuatro `for` anidados construyen todas las combinaciones
#   A<sub>x</sub>B<sub>y</sub> (metal, no metal, estequiometría).
# - `drop_duplicates("reduced")`: Ti1N1 y Ti2N2 son el mismo compuesto.
# - `scaler.transform` (¡sin `fit`!): los candidatos se escalan con las medias
#   del **entrenamiento**; el modelo solo entiende números en esa escala.
# - `df_cand.loc[X_cand.index, ...]`: escribe las predicciones solo en las
#   filas que sobrevivieron al `dropna()`.

# %% [markdown]
# **Filtrar los prometedores y verificar novedad** — ¿cuáles predice el modelo
# como super-rígidos (>300 GPa), y cuáles de esos ya conocemos?
#
# ⚠️ **Cuidado con contra qué se compara.** El caché de Materials Project de
# P2.2 solo tiene materiales con gap entre 0.3 y 2.5 eV: los carburos
# metálicos (gap 0) nunca están ahí. Por eso comparamos también contra el
# dataset de entrenamiento y, si existe, contra la lista completa de fórmulas
# de MP (`datos/mp_formulas_todas.json`, la genera `pre_cache_mp.py`).

# %%
top_candidates = (
    df_cand[df_cand["K_VRH_pred"] > 300]
    .sort_values("K_VRH_pred", ascending=False)
    .head(20)[["formula", "reduced", "K_VRH_pred"]]
    .copy()
)

formulas_entrenamiento = {Composition(f).reduced_formula for f in df["formula"]}
formulas_mp = set(df_mp["formula"])
lista_mp = DATOS / "mp_formulas_todas.json"
if lista_mp.exists():
    formulas_mp |= set(pd.read_json(lista_mp)["formula"])
else:
    print("ℹ Sin lista completa de MP: la columna en_MP solo mira el caché de gaps.")

top_candidates["en_entrenamiento"] = top_candidates["reduced"].isin(formulas_entrenamiento)
top_candidates["en_MP"] = top_candidates["reduced"].isin(formulas_mp)
print(top_candidates.to_string(index=False))

# 🤔 Los "conocidos" (WC, TaC...) validan el método: el modelo redescubre los
#    carburos duros clásicos. Los NO conocidos son hipótesis para verificar
#    con DFT o experimento — NO son descubrimientos todavía.

# %% [markdown]
# <div style="background:#FFF3E6;border-left:6px solid #E87C1E;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#C25E08;">🎯 Reto 2 &nbsp;·&nbsp; 10 min</b></div>
#
# **Reto:** amplía el pool de candidatos a **ternarios A-B-C**
# añadiendo un tercer conjunto de elementos de tu elección (¿otro metal?
# ¿otro no-metal?). Re-featuriza, re-predice.
#
# - ¿Aparece algún ternario con K_VRH predicho > 300 GPa?
# - ¿Confías más o menos en esas predicciones que en las de los binarios?
#   ¿Por qué? (Pista conceptual: ¿qué vio el modelo durante el entrenamiento?)

# %%
# [Su código aquí]

# %% [markdown]
# ⚠️ **Limitación importante** (la conexión con mañana): este método solo
# explora composiciones que *nosotros* enumeramos, dentro de la química que el
# modelo vio al entrenar. No puede proponer nada verdaderamente nuevo.
# Mañana usaremos modelos **generativos** que sí lo hacen.

# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>P2.4 — SHAP: ¿por qué el modelo predice lo que predice?</b> <span style="float:right;background:#E87C1E;color:white;padding:2px 12px;border-radius:12px;font-size:12px;font-weight:600;">⏱ 15 min</span></div>
#
# La importancia de features del RF dice *cuáles* importan globalmente, pero no
# *cómo* ni *para qué material*. **SHAP** asigna a cada feature su contribución
# exacta (en GPa) a cada predicción individual.

# %%
import shap

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test_sc)
print(f"Matriz SHAP: {shap_values.shape}  (materiales de test × features)")

# %% [markdown]
# <div style="background:#EEF3F9;border-left:6px solid #2E5E9B;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#2E5E9B;">🔍 Cómo funciona este código</b></div>
#
# - `TreeExplainer` calcula los valores SHAP de forma exacta y rápida para
#   modelos de árboles como el Random Forest.
# - `shap_values[i, j]` = cuántos GPa aporta el descriptor *j* a la predicción
#   del material *i*, respecto a la predicción promedio (`expected_value`).
# - Para cada material, la suma de sus valores SHAP + el promedio da
#   exactamente la predicción del modelo.

# %% [markdown]
# **Beeswarm (global):** cada punto es un material; el color es el valor del
# feature (rojo = alto), la posición horizontal es su efecto en la predicción.

# %%
shap.summary_plot(shap_values, X_test_sc, feature_names=feature_cols, max_display=15, show=False)
plt.tight_layout()
plt.show()

# Lectura: MeltingT alta (rojo) empuja K_VRH hacia arriba → consistente con
# la física del enlace. ¿Ven algún feature con efecto no monótono?

# %% [markdown]
# **Waterfall (individual):** ¿por qué el material más rígido del test set
# recibió esa predicción?

# %%
idx_max = int(np.argmax(y_test.values))
print(f"Material: {df.loc[y_test.index[idx_max], 'formula']} — K_VRH real: {y_test.values[idx_max]:.0f} GPa")

base = float(np.atleast_1d(explainer.expected_value)[0])
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[idx_max],
        base_values=base,
        data=X_test_sc[idx_max],
        feature_names=feature_cols,
    ),
    max_display=12,
)

# %% [markdown]
# ---
# ## 🏠 Ejercicio del Día 2 (para casa)
#
# Busca en el caché de Materials Project (o con la API si tienes key) los
# materiales con **bandgap entre 1.0 y 2.0 eV**:
#
# 1. ¿Cuántos cumplen? Haz un scatter de `band_gap` vs `Ef`.
# 2. Entrena un RandomForest que prediga `band_gap` desde features Magpie
#    (featuriza las fórmulas con `StrToComposition` + `ElementProperty`).
#    Reporta MAE y R².
# 3. ¿Cuál es el feature más importante? ¿Tiene sentido físico?
#
# *Pista: el flujo es idéntico al de P2.1, cambiando el dataset y el target.*
#
# ---
# ### 🔜 Conexión con el Día 3
# > El diseño inverso por filtrado solo encuentra lo que ya sabemos enumerar.
# > Mañana vamos más lejos: **generar** cristales nuevos, átomo por átomo, con
# > modelos de difusión — la misma familia de técnicas detrás de los
# > generadores de imágenes.
