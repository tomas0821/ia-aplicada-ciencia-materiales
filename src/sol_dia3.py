# %% [markdown]
# # ✅ Solución — Ejercicio del Día 3
#
# *Notebook del instructor.* Generación condicionada a Si-O-N + filtrado
# completo. En el servidor GPU usa MatterGen real; sin GPU usa el lote
# pre-generado `datos/mattergen_fallback/sion/`.


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
import torch
from pathlib import Path
from pymatgen.core import Structure

def encontrar_datos() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents[:3]]:
        if (base / "datos").is_dir():
            return base / "datos"
    raise FileNotFoundError("No encuentro la carpeta datos/")

DATOS = encontrar_datos()


# %%
# Misma función generar() del notebook 03 (GPU en Nukwa o lote pre-generado)
import os, sys, subprocess, shutil, zipfile

CURSO = DATOS.parent
MG_SRC = CURSO / "mattergen_src"
_HF_PROPIO = Path(f"/work/{os.environ.get('USER', '')}/hf_cache")
HF_CACHE = Path(os.environ.get("IA_CM_HF_HOME", _HF_PROPIO if _HF_PROPIO.is_dir() else "/work/trojas/hf_cache"))  # carpeta compartida del curso
SALIDAS = CURSO / "salidas_mattergen"

# cada condición usa un modelo ajustado distinto
MODELO_POR_CONDICION = {
    "chemical_system": "chemical_system",
    "space_group": "space_group",
    "dft_band_gap": "dft_band_gap",
}

requisitos = {
    "GPU": torch.cuda.is_available(),
    "código MatterGen (mattergen_src)": (MG_SRC / "mattergen").is_dir(),
    f"pesos en {HF_CACHE}": HF_CACHE.is_dir(),
}
for nombre, ok in requisitos.items():
    print(f"{'✓' if ok else '✗'} {nombre}")
MODO = "gpu" if all(requisitos.values()) else "fallback"
print(f"\nModo de trabajo: {MODO}")
if MODO == "fallback":
    print("  Usaremos estructuras pre-generadas de datos/mattergen_fallback/")

# %%
def leer_cifs_zip(ruta_zip):
    """Lee todas las estructuras de un zip de CIF sin descomprimirlo a disco."""
    with zipfile.ZipFile(ruta_zip) as z:
        return [Structure.from_str(z.read(n).decode(), fmt="cif")
                for n in sorted(z.namelist()) if n.endswith(".cif")]


def generar(lote: str, num_samples: int = 10, condicion=None, guia: float = 2.0):
    """Genera estructuras con MatterGen (GPU) o carga el lote pre-generado.

    lote      : nombre de la carpeta de fallback (incondicional, limnox, solar, cubico, sion)
    condicion : None o dict, p. ej. {"chemical_system": "Li-Mn-O"}
    guia      : factor de guía (qué tanto se obedece la condición)
    """
    if MODO == "gpu":
        modelo = "mattergen_base" if not condicion else MODELO_POR_CONDICION[next(iter(condicion))]
        salida = SALIDAS / lote
        shutil.rmtree(salida, ignore_errors=True)
        cmd = [sys.executable, "-m", "mattergen.scripts.generate", str(salida),
               f"--pretrained-name={modelo}", f"--batch_size={num_samples}", "--num_batches=1",
               "--record_trajectories=False"]  # sin trayectorias (~20 MB por lote)
        if condicion:
            cmd += [f"--properties_to_condition_on={condicion}",
                    f"--diffusion_guidance_factor={guia}"]
        env = dict(os.environ, HF_HOME=str(HF_CACHE), HF_HUB_OFFLINE="1",
                   PYTHONPATH=f"{MG_SRC}:{os.environ.get('PYTHONPATH', '')}")
        print(f"Generando {num_samples} estructuras con {modelo}... (1–5 min)")
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        zip_cif = salida / "generated_crystals_cif.zip"
        if r.returncode == 0 and zip_cif.exists():
            return leer_cifs_zip(zip_cif)
        print("⚠ MatterGen falló; uso el lote pre-generado. Últimas líneas del error:")
        print("\n".join(r.stderr.strip().splitlines()[-5:]))
    carpeta = DATOS / "mattergen_fallback" / lote
    cifs = sorted(carpeta.glob("*.cif"))[:num_samples]
    if not cifs:
        raise FileNotFoundError(f"No hay CIFs pre-generados en {carpeta}")
    return [Structure.from_file(str(f)) for f in cifs]


def cargar_lote(lote, n=20):
    """Reutiliza lo que generó el notebook 03 (salidas_mattergen/) o el fallback."""
    z = SALIDAS / lote / "generated_crystals_cif.zip"
    if z.exists():
        return leer_cifs_zip(z)[:n]
    cifs = sorted((DATOS / "mattergen_fallback" / lote).glob("*.cif"))[:n]
    return [Structure.from_file(str(f)) for f in cifs]

# %%
structs = generar("sion", num_samples=20, condicion={"chemical_system": "Si-O-N"})
print(f"{len(structs)} estructuras Si-O-N")

# %% [markdown]
# ## 1. Validez estructural (distancia mínima > 1.2 Å)
#
# Nota el umbral: 1.2 Å (enlace más corto razonable ~ C-H 1.09 Å; en
# inorgánicos Si-O ~1.6 Å). El 0.5 Å de la clase era un filtro grosero.

# %%
def min_dist(s):
    d = s.distance_matrix
    return float(d[d > 0].min())

df_gen = pd.DataFrame({
    "formula": [s.composition.reduced_formula for s in structs],
    "min_dist_A": [round(min_dist(s), 2) for s in structs],
    "space_group": [s.get_space_group_info()[0] for s in structs],
    "density": [round(s.density, 2) for s in structs],
})
df_gen["valida_1.2A"] = df_gen["min_dist_A"] > 1.2
print(df_gen.to_string(index=False))
print(f"\n1) Válidas (min_dist > 1.2 Å): {df_gen['valida_1.2A'].sum()}/{len(df_gen)}")

# %% [markdown]
# ## 2. Filtro SMACT de viabilidad iónica

# %%
from smact.screening import smact_validity

def is_smact_valid(struct) -> bool:
    try:
        return bool(smact_validity(struct.composition.reduced_formula,
                                   include_alloys=False))
    except Exception:
        return False

df_gen["smact_ok"] = [is_smact_valid(s) for s in structs]
print(f"2) Pasan SMACT: {df_gen['smact_ok'].sum()}/{len(df_gen)}")
print(df_gen[["formula", "valida_1.2A", "smact_ok"]].drop_duplicates("formula").to_string(index=False))

# %% [markdown]
# ## 3. Novedad contra Materials Project

# %%
import json
from pymatgen.core import Composition
mp_formulas = set()
mp_cache = DATOS / "mp_bandgap_cache.json"
if mp_cache.exists():
    mp_formulas |= set(pd.read_json(mp_cache)["formula"])
if (DATOS / "mp_formulas_todas.json").exists():
    mp_formulas |= set(pd.read_json(DATOS / "mp_formulas_todas.json")["formula"])
mp_formulas = {Composition(f).reduced_formula for f in mp_formulas}

df_gen["novedosa"] = ~df_gen["formula"].isin(mp_formulas)
supervivientes = df_gen[df_gen["valida_1.2A"] & df_gen["smact_ok"] & df_gen["novedosa"]]
print(f"3) Novedosas (no en el caché de MP): {df_gen['novedosa'].sum()}/{len(df_gen)}")
print(f"   Supervivientes de los 3 filtros: {len(supervivientes)}")
# Nota: sin datos/mp_formulas_todas.json el caché cubre solo bandgap 0.3-2.5 eV;
# SiO2 y Si3N4 (gaps > 4 eV) no están ahí y saldrían "novedosas" por error.
# En la prueba en Kabré salió Si2N2O (sinoita, mineral conocido): buen ejemplo
# para discutir por qué la novedad hay que verificarla contra MP completo.

# %% [markdown]
# ## 4. Las 3 mejores candidatas
#
# Criterio de "mejor" (justificable de varias formas): pasar los 3 filtros y
# ordenar por densidad razonable / simetría alta (más fácil de caracterizar
# por XRD y menos probable que sea un artefacto del generador).

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
from IPython.display import display

candidatas = supervivientes if len(supervivientes) else df_gen[df_gen["valida_1.2A"]]
top3_idx = candidatas.head(3).index
for i in top3_idx:
    titulo = f"{df_gen.loc[i,'formula']} ({df_gen.loc[i,'space_group']}) ρ={df_gen.loc[i,'density']} g/cm³"
    vista = ver_estructura(structs[i], titulo)
    if vista is not None:
        display(vista)

# Justificación tipo (lo que se espera del alumno):
# - Pasan validez geométrica, balance de carga y novedad.
# - Simetría alta → menos modos de fallo, XRD más limpia para confirmar fase.
# - Estequiometrías cercanas a fases conocidas (SiO2, Si3N4) → precursores y
#   rutas de síntesis (nitruración, sol-gel) ya establecidas.

# %% [markdown]
# ---
# # Retos en clase — Día 3
#
# ## Reto 1: criterio de validez propio
# Ejemplo con el criterio más físico: distancia mínima vs suma de radios
# atómicos de la pareja más cercana (un contacto MUY por debajo de la suma de
# radios es químicamente sospechoso).

# %%
from pymatgen.core import Element

def contacto_vs_radios(s, factor=0.6):
    """True si la distancia mínima supera factor*(suma de radios de la pareja)."""
    d = s.distance_matrix
    np.fill_diagonal(d, np.inf)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    r_i = Element(s[i].species_string).atomic_radius or 1.0
    r_j = Element(s[j].species_string).atomic_radius or 1.0
    return d[i, j] > factor * float(r_i + r_j), round(d[i, j], 2), round(float(r_i + r_j), 2)

lote_p32 = cargar_lote("incondicional", 10)   # las 10 estructuras de P3.2
for i, s in enumerate(lote_p32):
    ok, dmin, suma_r = contacto_vs_radios(s)
    marca = "✓" if ok else "❌"
    print(f"{s.composition.reduced_formula:8s} dmin={dmin} Å  suma_radios={suma_r} Å  {marca}")

# Con el lote real de Kabré las 10 pasan con factor 0.6: MatterGen rara vez
# produce contactos absurdos. Suba el factor a 0.9 para ver cuáles quedan cerca.
# Punto de discusión: este criterio ES relativo a la química (Si-O corto es
# normal; W-W a la misma distancia no) — por eso supera al umbral fijo.

# %% [markdown]
# ## Reto 2: diversidad de lotes
# Definición elegida (declararla es parte del reto): nº de fórmulas reducidas
# únicas y nº de grupos espaciales únicos, sobre el tamaño del lote.

# %%
def diversidad(lote):
    ss = cargar_lote(lote)
    formulas = {s.composition.reduced_formula for s in ss}
    grupos = {s.get_space_group_info()[0] for s in ss}
    return len(ss), len(formulas), len(grupos)

for lote in ["limnox", "cubico"]:
    n, nf, ng = diversidad(lote)
    print(f"{lote:8s}: {n} estructuras — {nf} fórmulas únicas, {ng} grupos espaciales únicos")

# Resultado con los lotes reales de Kabré: limnox 14 fórmulas y 7 grupos;
# cubico 20 fórmulas y 8 grupos. La condición de simetría restringe la
# estructura pero no la química (cualquier elemento vale), así que el lote
# cúbico es MÁS diverso en fórmulas; limnox solo puede combinar Li, Mn y O.
# Con los CIF placeholder (sin GPU ni lotes reales) los números no significan nada.
