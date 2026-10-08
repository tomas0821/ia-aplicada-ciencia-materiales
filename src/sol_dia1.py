# %% [markdown]
# # ✅ Solución — Ejercicio del Día 1
#
# *Notebook del instructor. Compartir con los alumnos después de la sesión
# del Día 2.*


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
from matminer.datasets import load_dataset

def encontrar_datos() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents[:3]]:
        if (base / "datos").is_dir():
            return base / "datos"
    raise FileNotFoundError("No encuentro la carpeta datos/")

DATOS = encontrar_datos()
import os
os.environ["MATMINER_DATA"] = str(DATOS / "matminer_cache")  # dataset incluido en el curso
df = load_dataset("elastic_tensor_2015")
print(df.shape)

# %% [markdown]
# ## 1. Los 5 materiales más anisótropos y sus sistemas cristalinos

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

top5 = df.nlargest(5, "elastic_anisotropy")[
    ["formula", "elastic_anisotropy", "space_group", "sistema"]
]
print(top5.to_string(index=False))

# Comentario: los más anisótropos suelen ser estructuras en capas o cadenas
# (baja simetría o enlaces muy direccionales) — NO cúbicos, donde la simetría
# fuerza isotropía elástica casi total.

# %% [markdown]
# ## 2. Featurizer Meredig: ¿cuántos features y cuál correlaciona más con G_VRH?

# %%
from matminer.featurizers.composition import Meredig
from matminer.featurizers.conversions import StrToComposition

stc = StrToComposition(target_col_id="composition")
import joblib
stc.set_n_jobs(joblib.cpu_count())  # solo los núcleos de esta sesión, no los 40 del nodo
df = stc.featurize_dataframe(df, "formula", ignore_errors=True, pbar=False)

mer = Meredig()
mer.set_n_jobs(joblib.cpu_count())  # solo los núcleos de esta sesión, no los 40 del nodo
print(f"Meredig genera {len(mer.feature_labels())} features")

df = mer.featurize_dataframe(df, col_id="composition", ignore_errors=True, pbar=False)

mer_cols = [c for c in mer.feature_labels() if c in df.columns]
corr_g = df[mer_cols + ["G_VRH"]].corr()["G_VRH"].drop("G_VRH")
print("\nTop 5 features Meredig por |correlación| con G_VRH:")
print(corr_g.abs().nlargest(5))

# %% [markdown]
# ## 3. Densidad con pymatgen y su correlación con K_VRH

# %%
# Para los 10 primeros materiales (como pide el enunciado)
df.loc[df.index[:10], "density_pymatgen"] = [
    s.density for s in df["structure"].iloc[:10]
]
print(df[["formula", "density_pymatgen"]].head(10).to_string(index=False))

# Para la correlación usamos TODOS los materiales (10 puntos serían muy pocos)
densidades = pd.Series([s.density for s in df["structure"]], index=df.index)
corr = densidades.corr(df["K_VRH"])
print(f"\nCorrelación densidad vs K_VRH (todo el dataset): {corr:.3f}")

plt.figure(figsize=(6, 5))
plt.scatter(densidades, df["K_VRH"], alpha=0.4, s=15)
plt.xlabel("Densidad pymatgen (g/cm³)")
plt.ylabel("K_VRH (GPa)")
plt.title(f"Densidad vs módulo de compresibilidad (r = {corr:.2f})")
plt.tight_layout()
plt.show()

# Comentario: correlación positiva moderada — materiales densos tienden a ser
# más rígidos (empaquetamiento + enlaces cortos), pero con mucha dispersión:
# la densidad sola NO basta para predecir K.

# %% [markdown]
# ---
# # Retos en clase — Día 1
#
# ## Reto 1: mayor K_VRH promedio vs mayor anisotropía promedio por sistema

# %%
resumen = df.groupby("sistema").agg(
    K_VRH_medio=("K_VRH", "mean"),
    anisotropia_media=("elastic_anisotropy", "mean"),
    n=("K_VRH", "size"),
).round(2)
print(resumen.sort_values("K_VRH_medio", ascending=False))

# NO son el mismo sistema. El cúbico tiene el mayor K promedio (142 GPa), pero
# la diferencia con tetragonal y ortorrómbico es pequeña (~136 GPa). En
# anisotropía, trigonal y ortorrómbico dominan (6.3 y 4.5) por los polimorfos
# de carbono en capas; cúbico (1.10) y tetragonal (1.01) quedan abajo: la
# simetría alta RESTRINGE cuán distinto responde el cristal según la dirección.
# Nota: el dataset no tiene materiales triclínicos.

# %% [markdown]
# ## Reto 2: el material más denso del dataset, visualizado

# %%
df["densidad"] = [s.density for s in df["structure"]]
idx_denso = df["densidad"].idxmax()
s_denso = df.loc[idx_denso, "structure"]
print(f"Más denso: {df.loc[idx_denso, 'formula']} — {df.loc[idx_denso, 'densidad']:.2f} g/cm³")
# Esperado: Os (~22.0 g/cm³), seguido de Ir (21.9) e Ir3W (21.3)

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


ver_estructura(s_denso)
