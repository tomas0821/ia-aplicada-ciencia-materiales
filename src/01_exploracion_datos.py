# %% [markdown]
# <div style="background:linear-gradient(90deg,#1A2E4A 0%,#2E5E9B 100%);color:white;padding:20px 26px;border-radius:12px;">
# <span style="font-size:12px;letter-spacing:3px;color:#D6E8F7;">CURSO IA APLICADA A CIENCIA DE MATERIALES &nbsp;·&nbsp; DÍA 1 · FUNDAMENTOS</span><br>
# <span style="font-size:26px;font-weight:700;">📊 Exploración de Datos de Materiales</span><br>
# <span style="font-size:14px;color:#D6E8F7;">Materiales → números: la base de todo lo que sigue</span>
# </div>
#
# | Bloque | Contenido | Tiempo |
# |--------|-----------|--------|
# | P1.1 | Setup y verificación del entorno | 20 min |
# | P1.2 | Carga y EDA del dataset | 30 min |
# | P1.3 | pymatgen: estructuras cristalinas | 30 min |
# | P1.4 | matminer: descriptores Magpie | 25 min |
# | — | Ejercicio del día | en casa |
#
# **La idea central de hoy:** para que un algoritmo aprenda de materiales,
# primero hay que convertir cada material en una lista de números
# (*features* o *descriptores*). Hoy exploramos un dataset real de propiedades
# elásticas y construimos esa representación numérica paso a paso.

# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>P1.1 — Setup y verificación del entorno</b> <span style="float:right;background:#E87C1E;color:white;padding:2px 12px;border-radius:12px;font-size:12px;font-weight:600;">⏱ 20 min</span></div>
#
# Ejecuta la celda siguiente. Deberías ver las versiones sin ningún error.
#
# > ⚠️ **¿Algo falló?** Levanta la mano ahora — no avances con errores de
# > entorno. Referencia: `00_test_entorno.ipynb` y el canal de soporte.


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
import seaborn as sns
import pymatgen.core
import matminer
import sklearn

print(f"numpy     {np.__version__}")
print(f"pandas    {pd.__version__}")
print(f"pymatgen  {pymatgen.core.__version__}")
print(f"matminer  {matminer.__version__}")
print(f"sklearn   {sklearn.__version__}")

# %% [markdown]
# La celda siguiente localiza la carpeta `datos/` del curso (funciona igual si
# ejecutas el notebook desde la raíz del repositorio o desde una subcarpeta):

# %%
from pathlib import Path

def encontrar_datos() -> Path:
    """Busca la carpeta datos/ en el directorio actual o sus padres."""
    for base in [Path.cwd(), *Path.cwd().parents[:3]]:
        candidata = base / "datos"
        if candidata.is_dir():
            return candidata
    raise FileNotFoundError("No encuentro la carpeta datos/ — revisa desde dónde abriste Jupyter")

DATOS = encontrar_datos()
print(f"Carpeta de datos: {DATOS}")

# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>P1.2 — Carga y EDA del dataset</b> <span style="float:right;background:#E87C1E;color:white;padding:2px 12px;border-radius:12px;font-size:12px;font-weight:600;">⏱ 30 min</span></div>
#
# Usamos `elastic_tensor_2015`: un dataset clásico con **propiedades elásticas
# calculadas por DFT** para 1,181 cristales inorgánicos
# (de Jong et al., *Scientific Data* 2015, vía Materials Project).
#
# Las dos propiedades que nos interesan hoy:
#
# - **K_VRH** — módulo de compresibilidad (*bulk modulus*, GPa): resistencia a
#   compresión uniforme. "¿Qué tan difícil es reducir su volumen?"
# - **G_VRH** — módulo de cizalla (*shear modulus*, GPa): resistencia a
#   deformación de forma. Correlaciona con la dureza.
#
# (VRH = promedio Voigt-Reuss-Hill de los límites superior e inferior.)

# %%
import os
os.environ["MATMINER_DATA"] = str(DATOS / "matminer_cache")
# ↳ el dataset viene incluido en datos/matminer_cache/ (no se descarga en clase)

from matminer.datasets import load_dataset

df = load_dataset("elastic_tensor_2015")

print(f"Filas y columnas: {df.shape}")
print(df.columns.tolist())

# %% [markdown]
# **Inspección inicial** — las tres preguntas de rigor ante cualquier dataset
# nuevo: ¿qué contiene?, ¿hay huecos?, ¿qué escala tienen los números?

# %%
df.head(3)

# %%
print("Valores nulos por columna (solo columnas con nulos):")
nulos = df.isnull().sum()
print(nulos[nulos > 0] if nulos.any() else "  ninguno ✓")

df[["K_VRH", "G_VRH", "elastic_anisotropy", "poisson_ratio"]].describe()

# %% [markdown]
# ### Visualización 1 — ¿Cómo se distribuyen las propiedades?
#
# Antes de modelar, *siempre* mirar las distribuciones: ¿son simétricas?
# ¿tienen colas largas? ¿valores imposibles?

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
df["K_VRH"].hist(bins=40, ax=axes[0], edgecolor="white")
axes[0].set(title="K_VRH (GPa)", xlabel="GPa", ylabel="N materiales")
df["G_VRH"].hist(bins=40, ax=axes[1], edgecolor="white")
axes[1].set(title="G_VRH (GPa)", xlabel="GPa")
df["elastic_anisotropy"].hist(bins=40, ax=axes[2], edgecolor="white")
axes[2].set(title="Anisotropía elástica", xlabel="adimensional")
plt.tight_layout()
plt.show()

# 🤔 Pregunta: ¿por qué la anisotropía tiene esa cola tan larga a la derecha?

# %% [markdown]
# ### Visualización 2 — ¿Qué sistemas cristalinos dominan el dataset?
#
# Este dataset **no** trae una columna `crystal_system`, pero sí el **número**
# del grupo espacial (1–230). Los rangos de números corresponden exactamente a
# los 7 sistemas cristalinos — una tabla que vale la pena conocer:
#
# | Grupo espacial | Sistema |
# |---------------|---------|
# | 1–2 | triclínico |
# | 3–15 | monoclínico |
# | 16–74 | ortorrómbico |
# | 75–142 | tetragonal |
# | 143–167 | trigonal |
# | 168–194 | hexagonal |
# | 195–230 | cúbico |

# %%
def sistema_cristalino(sg: int) -> str:
    if sg <= 2:    return "triclínico"
    if sg <= 15:   return "monoclínico"
    if sg <= 74:   return "ortorrómbico"
    if sg <= 142:  return "tetragonal"
    if sg <= 167:  return "trigonal"
    if sg <= 194:  return "hexagonal"
    return "cúbico"

df["sistema"] = df["space_group"].apply(sistema_cristalino)
# ↳ .apply llama a la función con cada valor de la columna y arma una columna nueva
orden = ["triclínico", "monoclínico", "ortorrómbico", "tetragonal",
         "trigonal", "hexagonal", "cúbico"]
df["sistema"].value_counts().reindex(orden).plot(kind="bar", figsize=(8, 4), edgecolor="white")
plt.title("Distribución por sistema cristalino (según nº de grupo espacial)")
plt.ylabel("Número de materiales")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Visualización 3 — Mapa de correlaciones
#
# ¿Qué propiedades "se mueven juntas"? El coeficiente de correlación de Pearson
# va de −1 (opuestas) a +1 (idénticas).

# %%
numeric_cols = ["K_VRH", "G_VRH", "elastic_anisotropy", "poisson_ratio", "nsites", "volume"]
plt.figure(figsize=(7, 5.5))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlaciones — elastic_tensor_2015")
plt.tight_layout()
plt.show()

# 🤔 K_VRH y G_VRH correlacionan fuerte (~0.8-0.95). ¿Por qué tiene sentido
#    físicamente? ¿Qué implicaría para un modelo que predice ambas?

# %% [markdown]
# ### Visualización 4 — K vs G, coloreado por tamaño de celda
#
# Con `USAR_PLOTLY = True` esta figura es **interactiva**: al pasar el mouse
# sobre un punto se ve qué material es. Encuentra el material de arriba a la derecha con pandas:
# `df.nlargest(3, "G_VRH")[["formula", "K_VRH", "G_VRH"]]`.

# %%
try:
    if not USAR_PLOTLY:
        raise ImportError("plotly desactivado en Kabré")
    import plotly.express as px

    figx = px.scatter(
        df, x="K_VRH", y="G_VRH", color="nsites",
        hover_name="formula",
        hover_data={"sistema": True, "K_VRH": ":.0f", "G_VRH": ":.0f", "nsites": True},
        color_continuous_scale="Viridis", opacity=0.6,
        labels={"K_VRH": "K_VRH (GPa)", "G_VRH": "G_VRH (GPa)", "nsites": "N sitios"},
        title="K vs G — pasa el mouse para ver el material",
    )
    figx.update_traces(marker=dict(size=6))
    figx.update_layout(width=750, height=520, font_color="#1A2E4A",
                       plot_bgcolor="white", title_font_weight=700)
    figx.update_xaxes(gridcolor="#E8EAEE")
    figx.update_yaxes(gridcolor="#E8EAEE")
    figx.show()
except ImportError:
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(df["K_VRH"], df["G_VRH"], c=df["nsites"], cmap="viridis", alpha=0.5, s=20)
    plt.colorbar(sc, label="N sitios en la celda")
    plt.xlabel("K_VRH (GPa)")
    plt.ylabel("G_VRH (GPa)")
    plt.title("K vs G — elastic_tensor_2015")
    plt.show()

# %% [markdown]
# <div style="background:#FFF3E6;border-left:6px solid #E87C1E;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#C25E08;">🎯 Reto 1 &nbsp;·&nbsp; 5 min</b></div>
#
# **Reto:** ¿qué sistema cristalino tiene el mayor K_VRH
# *promedio*? ¿Y cuál es el más *anisótropo* en promedio? ¿Es el mismo?
# ¿Tiene sentido físico que (no) coincidan?
#
# *Pista única: `groupby`. El resto es investigación suya.*

# %%
# [Su código aquí]


# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>P1.3 — pymatgen: estructuras cristalinas</b> <span style="float:right;background:#E87C1E;color:white;padding:2px 12px;border-radius:12px;font-size:12px;font-weight:600;">⏱ 30 min</span></div>
#
# `pymatgen` es la navaja suiza de la cristalografía computacional. Su objeto
# central es `Structure`: red + posiciones atómicas + especies.
#
# En este dataset, la columna `structure` ya contiene objetos `Structure`
# listos para usar. (Si tuvieras un JSON/dict, se reconstruye con
# `Structure.from_dict(...)`.)

# %%
from pymatgen.core import Structure, Composition, Element

struct = df["structure"].iloc[0]
print(type(struct))

print(f"Fórmula:           {struct.formula}")
print(f"Volumen:           {struct.volume:.2f} Å³")
print(f"Densidad:          {struct.density:.3f} g/cm³")
print(f"N sitios:          {struct.num_sites}")
print(f"Grupo espacial:    {struct.get_space_group_info()}")
print(f"Parámetros de red: a={struct.lattice.a:.3f}  b={struct.lattice.b:.3f}  c={struct.lattice.c:.3f} Å")

# %% [markdown]
# **Vecinos más cercanos** — la base de casi cualquier análisis local
# (coordinación, enlaces, descriptores estructurales):

# %%
neighbors = struct.get_neighbors(struct[0], r=3.0)
print(f"El sitio 0 ({struct[0].species_string}) tiene {len(neighbors)} vecinos en r < 3 Å")
for n in neighbors[:5]:
    print(f"  {n.species_string:6s} a {n.nn_distance:.3f} Å")

# %% [markdown]
# <div style="background:#EEF3F9;border-left:6px solid #2E5E9B;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#2E5E9B;">🔍 Cómo funciona este código</b></div>
#
# - `struct[0]` es el primer **sitio** (átomo) de la celda.
# - `get_neighbors` busca átomos a menos de `r` Å **incluyendo las celdas
#   vecinas**: el cristal es infinito y periódico, así que un vecino puede
#   estar "del otro lado" de la celda. Por eso a veces aparecen más vecinos
#   de los que hay átomos en la celda.

# %% [markdown]
# ### Visualización de la estructura
#
# `ver_estructura` usa el visor 3D `nglview` (se rota con el mouse) cuando está
# disponible, y si no, dibuja dos vistas estáticas con ASE. Compara al menos dos estructuras de
# sistemas cristalinos distintos.

# %%
def ver_estructura(struct, titulo=None):
    """Muestra una estructura: visor 3D (nglview) si está disponible,
    o dos vistas estáticas dibujadas con ASE si no lo está."""
    if USAR_NGLVIEW:
        import nglview as nv
        view = nv.show_pymatgen(struct)
        view.add_unitcell()
        return view
    from ase.visualize.plot import plot_atoms
    from pymatgen.io.ase import AseAtomsAdaptor
    atoms = AseAtomsAdaptor.get_atoms(struct)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, rot, lab in zip(axes, ["0x,0y,0z", "-80x,-15y,0z"], ["vista superior", "vista lateral"]):
        plot_atoms(atoms, ax, rotation=rot, radii=0.4, show_unit_cell=2)
        ax.set_title(lab, fontsize=10)
        ax.axis("off")
    fig.suptitle(titulo or struct.composition.reduced_formula, fontweight="bold")
    plt.show()


# %%
ver_estructura(struct)

# %%
# Segunda estructura, de otro sistema cristalino, para comparar visualmente
struct2 = df["structure"].iloc[100]
print(f"{struct2.formula} — grupo espacial {struct2.get_space_group_info()}")
ver_estructura(struct2)

# %% [markdown]
# ### Composición y propiedades elementales
#
# `Composition` entiende la fórmula química; `Element` da acceso a la tabla
# periódica completa. Estos son los ladrillos de los descriptores de mañana.

# %%
comp = Composition(df["formula"].iloc[5])
print(f"Fórmula:            {comp}")
print(f"Fracción atómica:   {dict(comp.fractional_composition.as_dict())}")
print(f"Peso molecular:     {comp.weight:.2f} uma")
print(f"Fórmula reducida:   {comp.get_reduced_formula_and_factor()}")

elem_fe = Element("Fe")
print(f"\nRadio atómico Fe:    {elem_fe.atomic_radius} Å")
print(f"Electronegatividad:  {elem_fe.X}")

# %% [markdown]
# <div style="background:#FFF3E6;border-left:6px solid #E87C1E;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#C25E08;">🎯 Reto 2 &nbsp;·&nbsp; 10 min</b></div>
#
# **Reto:** encuentra el material **más denso** de todo el
# dataset y visualízalo en 3D con su celda unitaria.
#
# Sin pistas de código: ya usaron todo lo necesario en esta sección.
# (Verificación: la densidad ganadora supera los 20 g/cm³.)

# %%
# [Su código aquí]


# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>P1.4 — matminer: descriptores Magpie</b> <span style="float:right;background:#E87C1E;color:white;padding:2px 12px;border-radius:12px;font-size:12px;font-weight:600;">⏱ 25 min</span></div>
#
# **El paso conceptual más importante del día.** Un modelo de ML no puede leer
# "Fe₂O₃" — necesita números. Los descriptores **Magpie** (Ward et al. 2016)
# convierten cualquier composición en **132 números**: estadísticas
# (media, rango, desviación...) de propiedades elementales (electronegatividad,
# radio, temperatura de fusión, electrones de valencia...).
#
# Material → composición → 132 features → (mañana) modelo de ML

# %%
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition

# Paso 1: convertir strings de fórmula a objetos Composition
stc = StrToComposition(target_col_id="composition")
import joblib
stc.set_n_jobs(joblib.cpu_count())  # solo los núcleos de esta sesión, no los 40 del nodo
df = stc.featurize_dataframe(df, "formula", ignore_errors=True, pbar=False)

# Paso 2: calcular los descriptores Magpie
ep = ElementProperty.from_preset("magpie")
ep.set_n_jobs(joblib.cpu_count())  # solo los núcleos de esta sesión, no los 40 del nodo
print(f"Número de features que genera: {len(ep.feature_labels())}")
print(f"Primeras 5: {ep.feature_labels()[:5]}")

# %%
df = ep.featurize_dataframe(df, col_id="composition", ignore_errors=True, pbar=False)
print(f"Shape final del DataFrame: {df.shape}")

feature_cols = [c for c in df.columns if "MagpieData" in c]
print(f"Total features Magpie: {len(feature_cols)}")

# %% [markdown]
# <div style="background:#EEF3F9;border-left:6px solid #2E5E9B;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#2E5E9B;">🔍 Cómo funciona este código</b></div>
#
# - Todos los *featurizers* de matminer siguen el mismo patrón:
#   `featurize_dataframe(df, col_id=...)` lee una columna y **agrega** columnas
#   nuevas al DataFrame.
# - `StrToComposition` convierte el texto `"Fe2O3"` en un objeto `Composition`
#   (sabe cuántos átomos de cada elemento hay).
# - `ElementProperty` calcula, para cada una de 22 propiedades elementales,
#   6 estadísticas ponderadas por la fracción atómica: 22 × 6 = 132 columnas.
# - `ignore_errors=True` deja `NaN` si una fórmula falla, en vez de detener
#   todo. Mañana quitamos esas filas con `dropna()`.

# %% [markdown]
# ### ¿Qué features "ven" la física?
#
# Correlación de cada feature con K_VRH — un primer vistazo (¡no definitivo!)
# de qué información importa:

# %%
corr = df[feature_cols + ["K_VRH"]].corr()["K_VRH"].drop("K_VRH")
top10 = corr.abs().nlargest(10)

top10.sort_values().plot(kind="barh", figsize=(8, 5), edgecolor="white")
plt.title("Top 10 features Magpie más correlacionadas con K_VRH")
plt.xlabel("|correlación de Pearson|")
plt.tight_layout()
plt.show()

print(top10)

# 🤔 La temperatura de fusión promedio (mean MeltingT) suele salir arriba.
#    ¿Por qué un material que funde a alta temperatura tiende a ser rígido?
#    Pista: ambas cosas dependen de la fuerza del enlace.

# %% [markdown]
# ### Guardar el trabajo para el Día 2
#
# Mañana entrenaremos modelos sobre estas features — no queremos recalcularlas.

# %%
df.to_pickle(DATOS / "df_con_features_dia1.pkl")
print(f"Guardado en {DATOS/'df_con_features_dia1.pkl'}")
print("Mañana: df = pd.read_pickle(DATOS / 'df_con_features_dia1.pkl')")

# %% [markdown]
# ---
# ## 🏠 Ejercicio del Día 1 (para casa)
#
# Usando `elastic_tensor_2015` y lo aprendido hoy:
#
# 1. ¿Cuáles son los **5 materiales con mayor `elastic_anisotropy`**? ¿A qué
#    grupos espaciales pertenecen?
# 2. Añade el featurizer **`Meredig`** de matminer
#    (`from matminer.featurizers.composition import Meredig`). ¿Cuántos features
#    genera? ¿Cuál correlaciona más con `G_VRH`?
# 3. Calcula la **densidad** de los 10 primeros materiales con pymatgen
#    (`struct.density`) y añádela como columna `density_pymatgen`.
#    ¿Correlaciona con `K_VRH`?
#
# *Soluciones mañana al inicio de la sesión.*
#
# ---
# ### 🔜 Conexión con el Día 2
# > Ya tienen los materiales representados como números. Mañana les enseñamos a
# > **predecir propiedades que no han medido** — y a buscar materiales con las
# > propiedades que desean.
