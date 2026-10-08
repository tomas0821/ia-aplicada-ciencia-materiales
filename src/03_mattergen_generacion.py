# %% [markdown]
# <div style="background:linear-gradient(90deg,#1A2E4A 0%,#2E5E9B 100%);color:white;padding:20px 26px;border-radius:12px;">
# <span style="font-size:12px;letter-spacing:3px;color:#D6E8F7;">CURSO IA APLICADA A CIENCIA DE MATERIALES &nbsp;·&nbsp; DÍA 3 · GENERACIÓN (GPU)</span><br>
# <span style="font-size:26px;font-weight:700;">🧬 Generación de Materiales con MatterGen</span><br>
# <span style="font-size:14px;color:#D6E8F7;">Cristales nuevos átomo por átomo, con modelos de difusión</span>
# </div>
#
# | Bloque | Contenido | Tiempo |
# |--------|-----------|--------|
# | P3.1 | Sesión GPU en Kabré y setup de MatterGen | 15 min |
# | P3.2 | Generación incondicional y auditoría | 25 min |
# | P3.3 | Generación condicional | 35 min |
# | P3.4 | Filtrado de candidatos | 15 min |
# | R3 | Retos y cierre | 20 min |
#
# **La idea central de hoy:** ayer *buscamos* materiales en listas que nosotros
# enumeramos. Hoy un modelo de difusión (**MatterGen**, Zeni et al., *Nature*
# 2025) **genera cristales nuevos átomo por átomo**: parte de ruido y lo
# "des-ruidiza" hasta obtener una estructura — la misma familia de técnicas
# que genera imágenes, aplicada a redes cristalinas.
#
# > ⚠️ **Plan B integrado:** si la GPU o el modelo no están disponibles, este
# > notebook carga automáticamente estructuras **pre-generadas** desde
# > `datos/mattergen_fallback/` y todo el análisis funciona igual.

# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>P3.1 — Setup</b> <span style="float:right;background:#E87C1E;color:white;padding:2px 12px;border-radius:12px;font-size:12px;font-weight:600;">⏱ 15 min</span></div>
#
# Esta sesión corre en la cola **kura** (CPU), como los demás días; la GPU (nukwa) solo recibe los trabajos cortos de MatterGen de la tarea.
# Primero, revisemos qué tenemos disponible:


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

print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

def encontrar_datos() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents[:3]]:
        if (base / "datos").is_dir():
            return base / "datos"
    raise FileNotFoundError("No encuentro la carpeta datos/")

DATOS = encontrar_datos()

# %% [markdown]
# **Preparar MatterGen.** MatterGen es un programa de **línea de comandos**
# (`mattergen-generate`): no se importa como una función de Python. Para usarlo
# desde el notebook necesitamos tres cosas:
#
# 1. **GPU** (en clase no hay: Jupyter corre en kura y se usa el modo fallback; la GPU se usa en la tarea).
# 2. **Código fuente** de MatterGen v1.0.3 en `curso/mattergen_src/`: la
#    instalación del kernel no trae algunos archivos de configuración, así que
#    usamos esta copia.
# 3. **Pesos** de los modelos (~2 GB) en la carpeta compartida del curso, `/work/trojas/hf_cache`,
#    para no descargarlos en clase.
#
# Si falta cualquiera, el notebook pasa a **modo fallback** y usa estructuras
# que MatterGen generó en Kabré durante la preparación del curso.

# %%
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

# %% [markdown]
# <div style="background:#EEF3F9;border-left:6px solid #2E5E9B;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#2E5E9B;">🔍 Cómo funciona este código</b></div>
#
# - `subprocess.run(cmd, env=env)` ejecuta un programa externo como si lo
#   escribiéramos en la terminal. `sys.executable` es el Python del kernel.
# - `env`: copia de las variables de entorno con tres cambios:
#   `HF_HOME` (dónde están los pesos), `HF_HUB_OFFLINE=1` (no descargar nada) y
#   `PYTHONPATH` (usar primero el código de `mattergen_src`).
# - La condición se pasa como texto: `str({"chemical_system": "Li-Mn-O"})`
#   produce exactamente el formato que espera MatterGen.
# - MatterGen escribe un zip con un CIF por estructura; `leer_cifs_zip` los
#   convierte en objetos `Structure` de pymatgen.

# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>P3.2 — Generación incondicional</b> <span style="float:right;background:#E87C1E;color:white;padding:2px 12px;border-radius:12px;font-size:12px;font-weight:600;">⏱ 25 min</span></div>
#
# Sin ninguna restricción: "MatterGen, dame 10 cristales que te parezcan
# físicamente razonables". Luego auditamos qué tan razonables son.

# %%
structures_raw = generar("incondicional", num_samples=10)
print(f"{len(structures_raw)} estructuras obtenidas")

# %% [markdown]
# **Auditoría básica de validez.** Un generador puede producir átomos
# absurdamente cerca (estructura no física). Regla mínima: ninguna distancia
# interatómica < 0.5 Å. (Filtros serios añaden balance de carga, energía, etc.)

# %%
results = []
for i, struct in enumerate(structures_raw):
    try:
        dists = struct.distance_matrix
        min_dist = float(dists[dists > 0].min())
        valid = min_dist > 0.5
    except Exception:
        min_dist, valid = 0.0, False

    results.append({
        "idx": i,
        "formula": struct.composition.reduced_formula,
        "nsites": struct.num_sites,
        "volume_A3": round(struct.volume, 1),
        "density": round(struct.density, 2),
        "space_group": struct.get_space_group_info()[0],
        "min_dist_A": round(min_dist, 2),
        "valid": valid,
    })

df_gen = pd.DataFrame(results)
print(df_gen.to_string(index=False))
print(f"\nEstructuras válidas: {df_gen['valid'].sum()}/{len(df_gen)}")

# %% [markdown]
# <div style="background:#EEF3F9;border-left:6px solid #2E5E9B;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#2E5E9B;">🔍 Cómo funciona este código</b></div>
#
# - `struct.distance_matrix` es una matriz N×N con la distancia entre cada par
#   de átomos de la celda, respetando las condiciones periódicas (usa la imagen
#   más cercana). La diagonal vale 0, por eso filtramos `dists > 0`.
# - `get_space_group_info()` usa spglib con una tolerancia `symprec=0.01` Å por
#   defecto. Una estructura generada casi simétrica puede salir P1 con esa
#   tolerancia y cúbica con `symprec=0.1`: la simetría detectada depende de la
#   tolerancia que elijamos.
# - `struct.density` sale en g/cm³ a partir de masas atómicas y volumen de celda.

# %% [markdown]
# **Visualización 3D** de las primeras 3 válidas — ¿se *ven* como cristales?

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

for idx, row in df_gen[df_gen["valid"]].head(3).iterrows():
    struct = structures_raw[row["idx"]]
    vista = ver_estructura(struct, f"{row['formula']} ({row['space_group']})")
    if vista is not None:
        display(vista)

# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>P3.3 — Generación condicional</b> <span style="float:right;background:#E87C1E;color:white;padding:2px 12px;border-radius:12px;font-size:12px;font-weight:600;">⏱ 35 min</span></div>
#
# El verdadero poder de MatterGen: **pedirle materiales con requisitos**.
# Tres condiciones típicas de diseño:
#
# | Caso | Condición | Motivación |
# |------|-----------|-----------|
# | (a) | `chemical_system = "Li-Mn-O"` | Cátodos de batería sin Co/Ni |
# | (b) | `dft_band_gap = 1.5` eV | Cerca del óptimo Shockley-Queisser para solar |
# | (c) | `space_group = 225` (Fm-3m, cúbico) | Propiedades isótropas |
#
# Cada condición usa un modelo ajustado distinto (`chemical_system`,
# `dft_band_gap`, `space_group`). El factor de guía (`guia=2.0`) controla qué
# tanto se obedece la condición: más alto obedece más pero reduce diversidad.

# %%
# --- (a) Condición composicional: Li-Mn-O ---
structs_limnox = generar("limnox", num_samples=20,
                         condicion={"chemical_system": "Li-Mn-O"})
formulas_a = sorted({s.composition.reduced_formula for s in structs_limnox})
print(f"Fórmulas Li-Mn-O generadas ({len(structs_limnox)} estructuras):")
print(formulas_a)

# %%
# --- (b) Condición de propiedad: bandgap 1.4-1.6 eV ---
structs_solar = generar("solar", num_samples=20,
                        condicion={"dft_band_gap": 1.5})
print(f"Candidatos solares: {len(structs_solar)}")
print(sorted({s.composition.reduced_formula for s in structs_solar}))
# Verificación independiente del bandgap: requiere un predictor externo
# (p.ej. CGCNN pre-entrenado) — discusión en clase, no lo corremos hoy.

# %%
# --- (c) Condición de simetría: sistema cúbico ---
structs_cubic = generar("cubico", num_samples=20,
                        condicion={"space_group": 225})  # Fm-3m, grupo cúbico (tipo NaCl)

# ¿Qué fracción cumple realmente la condición? (verificamos con pymatgen)
# symprec=0.1 Å: tolerancia más laxa, porque lo generado nunca es perfectamente simétrico
info = [s.get_space_group_info(symprec=0.1) for s in structs_cubic]
n_cubic = sum(1 for _, num in info if num >= 195)  # grupos 195-230 = cúbicos
print(f"Grupos espaciales obtenidos: {sorted({sym for sym, _ in info})}")
n_225 = sum(1 for _, num in info if num == 225)
print(f"Cumplen condición cúbica: {n_cubic}/{len(structs_cubic)}")
print(f"Exactamente grupo 225:    {n_225}/{len(structs_cubic)}")

# 🤔 En la prueba en Kabré (setiembre 2026) salieron 16/20 cúbicas y solo 7/20
#    exactamente en el grupo 225. La condición GUÍA la
#    difusión, no la garantiza — igual que un prompt no garantiza la imagen.
#    Por eso el paso siguiente (filtrar y verificar) nunca es opcional.

# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>P3.4 — Filtrado de candidatos</b> <span style="float:right;background:#E87C1E;color:white;padding:2px 12px;border-radius:12px;font-size:12px;font-weight:600;">⏱ 15 min</span></div>
#
# Generar es barato; sintetizar es caro. Antes de proponer un candidato al
# laboratorio se filtra por:
#
# 1. **Viabilidad química** — ¿existe una combinación de estados de oxidación
#    que balancee la carga? (SMACT)
# 2. **Novedad** — ¿ya existe en Materials Project?
# 3. **Estabilidad (aproximada)** — ¿qué tan lejos queda del casco convexo?
#    (potencial interatómico universal MatterSim, más abajo)

# %%
import json
from smact.screening import smact_validity

def is_smact_valid(struct) -> bool:
    """¿Existe una asignación de estados de oxidación que balancee la carga
    y respete la electronegatividad (Pauling)?"""
    try:
        return bool(smact_validity(struct.composition.reduced_formula,
                                   include_alloys=False))
    except Exception:
        return False

# Novedad: fórmulas conocidas = caché MP del Día 2 + lista completa de MP
# (datos/mp_formulas_todas.json, si se descargó con pre_cache_mp.py)
# + fórmulas del dataset de entrenamiento del Día 1.
from pymatgen.core import Composition
mp_formulas = set()
mp_cache = DATOS / "mp_bandgap_cache.json"
if mp_cache.exists():
    mp_formulas |= set(pd.read_json(mp_cache)["formula"])
if (DATOS / "mp_formulas_todas.json").exists():
    mp_formulas |= set(pd.read_json(DATOS / "mp_formulas_todas.json")["formula"])
if (DATOS / "df_con_features_dia1.pkl").exists():
    mp_formulas |= set(pd.read_pickle(DATOS / "df_con_features_dia1.pkl")["formula"])
import gzip
REF_MP = DATOS / "hull" / "referencia_mp_eform.json.gz"   # ~95 000 fórmulas de Materials Project
if REF_MP.exists():
    mp_formulas |= set(json.load(gzip.open(REF_MP, "rt")))
mp_formulas = {Composition(f).reduced_formula for f in mp_formulas}
print(f"Fórmulas conocidas para comparar novedad: {len(mp_formulas)}")
if not (DATOS / "mp_formulas_todas.json").exists() and not REF_MP.exists():
    print("  (sin la lista completa de MP: 'novedoso' aquí es solo una aproximación)")

# %% [markdown]
# <div style="background:#EEF3F9;border-left:6px solid #2E5E9B;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#2E5E9B;">🔍 Cómo funciona este código</b></div>
#
# - `smact_validity` prueba todas las combinaciones de estados de oxidación
#   conocidos de cada elemento y devuelve `True` si alguna suma cero carga
#   neta **y** el elemento más electronegativo lleva la carga negativa.
#   `include_alloys=False` evita aprobar automáticamente las mezclas de metales.
# - La novedad se decide por **fórmula reducida** (`Composition.reduced_formula`,
#   p. ej. Li4Mn4O8 → LiMnO2). Es un criterio grueso: un polimorfo nuevo de una
#   fórmula conocida contaría como "no novedoso". Para comparar estructuras de
#   verdad se usa `StructureMatcher` de pymatgen.

# %%
filtered = []
for struct in structs_limnox:
    if not is_smact_valid(struct):
        continue
    if struct.composition.reduced_formula in mp_formulas:
        continue  # ya conocido
    filtered.append(struct)

print(f"Candidatos Li-Mn-O tras filtrado: {len(filtered)}/{len(structs_limnox)}")
for s in filtered[:5]:
    print(f"  {s.composition.reduced_formula:12s} — {s.get_space_group_info()[0]:10s} — ρ={s.density:.2f} g/cm³")

# 🤔 En la prueba en Kabré pasaron solo 2/20. Varias estructuras eran LiMnO2 y
#    Li2MnO3, dos cátodos muy conocidos: sin la lista completa de Materials
#    Project habrían salido "novedosos". Moraleja: la novedad se verifica contra
#    una base de datos completa.
# ⚠️ "Pasa los filtros" ≠ "sintetizable". El veredicto final siempre es
#    DFT + experimento. Estos filtros solo ordenan la cola de candidatos.

# %% [markdown]
# **3. Estabilidad aproximada con un potencial universal.** SMACT dice si la
# fórmula *puede* balancear cargas, pero no si el cristal es estable. Para eso
# se usa la energía sobre el casco convexo (E_hull, ver la presentación del
# inicio). Calcularla con DFT toma horas por estructura; un **potencial
# interatómico universal** (MatterSim, de Microsoft, entrenado con millones de
# cálculos DFT) lo aproxima en segundos:
#
# 1. relajamos cada candidato con MatterSim (posiciones y celda),
# 2. calculamos su energía de formación y la comparamos con ~95 000 compuestos
#    de Materials Project del mismo sistema químico.
#
# Regla práctica: **E_hull < 0.1 eV/átomo** = candidato razonable (muchos
# materiales útiles son metaestables). El error típico de este atajo es de
# ~0.05 eV/átomo: sirve para **ordenar** candidatos, no reemplaza a DFT.

# %%
import itertools
import ase.constraints, ase.filters, ase.stress
for _n in ["ExpCellFilter", "UnitCellFilter", "StrainFilter", "FrechetCellFilter"]:
    setattr(ase.constraints, _n, getattr(ase.filters, _n))    # MatterSim 1.1 con ASE 3.29
ase.constraints.full_3x3_to_voigt_6_stress = ase.stress.full_3x3_to_voigt_6_stress
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from mattersim.forcefield import MatterSimCalculator

HULL = DATOS / "hull"
INFO_MU = json.load(open(HULL / "mu_mattersim.json"))
MU, U_EL = INFO_MU["mu"], INFO_MU["U_EL"]
REF_SIS = {}                      # sistema químico -> [(composición, e_form DFT)]
for _f, _e in json.load(gzip.open(HULL / "referencia_mp_eform.json.gz", "rt")).items():
    _c = Composition(_f)
    REF_SIS.setdefault(tuple(sorted(str(x) for x in _c.elements)), []).append((_c, _e))
calc_ms = MatterSimCalculator(load_path=str(HULL / "mattersim-v1.0.0-1M.pth"),
                              device="cuda" if torch.cuda.is_available() else "cpu")


def relajar(struct, pasos=200):
    """Relaja posiciones y celda con MatterSim. Devuelve (estructura, energía total en eV)."""
    atoms = AseAtomsAdaptor.get_atoms(struct)
    atoms.calc = calc_ms
    FIRE(FrechetCellFilter(atoms), logfile=None).run(fmax=0.05, steps=pasos)
    return AseAtomsAdaptor.get_structure(atoms), atoms.get_potential_energy()


def e_form_mattersim(comp, energia):
    """Energía de formación (eV/átomo) en la escala de Materials Project."""
    d = comp.get_el_amt_dict()
    ref = sum(n * MU[el] for el, n in d.items())
    if "O" in d or "F" in d:                      # corrección GGA+U de Materials Project
        ref += sum(d[el] * MU[f"U_{el}"] for el in U_EL if el in d)
    return (energia - ref) / comp.num_atoms


def e_hull(comp, e_form):
    """Distancia al casco convexo formado por los compuestos conocidos del sistema."""
    els = sorted(str(x) for x in comp.elements)
    entradas = [PDEntry(Composition(x), 0.0) for x in els]
    for r in range(1, len(els) + 1):
        for sub in itertools.combinations(els, r):
            entradas += [PDEntry(c, e * c.num_atoms) for c, e in REF_SIS.get(sub, [])]
    return PhaseDiagram(entradas).get_e_above_hull(PDEntry(comp, e_form * comp.num_atoms),
                                                   allow_negative=True)


def evaluar_estabilidad(structs):
    filas = []
    for s in structs:
        if any(str(el) not in MU for el in s.composition.elements):
            continue                                   # elemento sin referencia
        s_rel, energia = relajar(s)
        ef = e_form_mattersim(s_rel.composition, energia)
        filas.append({"fórmula": s.composition.reduced_formula,
                      "grupo": s_rel.get_space_group_info(symprec=0.1)[0],
                      "ΔV_%": round(100 * (s_rel.volume / s.volume - 1), 1),
                      "e_form": round(ef, 3), "E_hull": round(e_hull(s_rel.composition, ef), 3),
                      "SMACT": is_smact_valid(s)})
    return pd.DataFrame(filas).sort_values("E_hull").reset_index(drop=True)

# %%
import time
t0 = time.time()
estab = evaluar_estabilidad(structs_limnox)
print(estab.to_string())
print(f"\nE_hull < 0.1 eV/átomo: {(estab.E_hull < 0.1).sum()}/{len(estab)}   ({time.time() - t0:.0f} s)")

# 🤔 En la prueba en Kabré (4 núcleos, unos 3 min): 16/20 con E_hull < 0.1 y
#    LiMnO2 justo sobre el casco (E_hull ≈ 0), como debe ser para un compuesto
#    estable conocido. Fíjense en los que SMACT rechaza pero tienen E_hull bajo,
#    y al revés: ¿qué filtro creerían más y por qué?

# %% [markdown]
# <div style="background:#EEF3F9;border-left:6px solid #2E5E9B;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#2E5E9B;">🔍 Cómo funciona este código</b></div>
#
# - `MatterSimCalculator` es una **red neuronal que reemplaza a DFT**: recibe
#   las posiciones de los átomos y devuelve energía y fuerzas en milisegundos.
# - `FIRE(FrechetCellFilter(atoms))` mueve átomos **y** celda hasta que las
#   fuerzas son menores que 0.05 eV/Å, igual que una relajación DFT.
# - La energía total de MatterSim no se puede comparar directamente con
#   Materials Project; `MU` son "energías de referencia por elemento" que se
#   ajustaron una vez (con ~7 600 estructuras de MP) para que las dos escalas
#   coincidan. Error de ese ajuste: 0.05 eV/átomo en promedio; con él, 98 % de
#   los compuestos estables de MP quedan con E_hull < 0.1 eV/átomo.
# - `PhaseDiagram` de pymatgen construye el casco convexo con los compuestos
#   conocidos del mismo sistema químico y mide qué tan arriba queda el nuestro.
#   E_hull negativo = "más estable que todo lo conocido": o es un hallazgo, o
#   (más probable) es error del modelo. Ahí es donde se corre DFT.

# %% [markdown]
# <div style="background:#FFF3E6;border-left:6px solid #E87C1E;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#C25E08;">🎯 Reto 1 &nbsp;·&nbsp; 10 min</b></div>
#
# **Reto:** el filtro de distancia mínima (0.5 Å) es
# deliberadamente laxo. Diseña **tu propio criterio de validez adicional** y
# aplícalo a las 10 estructuras de P3.2. Ideas para investigar (elige UNA y
# justifícala): densidad físicamente razonable, número de especies químicas,
# volumen por átomo, o comparar la distancia mínima contra la **suma de radios
# atómicos** de la pareja más cercana (pymatgen tiene todo lo necesario).
#
# ¿Tu criterio descarta estructuras que el filtro de 0.5 Å dejó pasar?

# %%
# [Su código aquí]

# %% [markdown]
# <div style="background:#FFF3E6;border-left:6px solid #E87C1E;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#C25E08;">🎯 Reto 2 &nbsp;·&nbsp; 5 min</b></div>
#
# **Reto:** compara los lotes `limnox` y `cubico`: ¿cuál tiene
# mayor **diversidad** de fórmulas y de grupos espaciales? Define cómo medir
# "diversidad" (esa decisión también es parte del reto) y escríbelo en una
# frase antes de programarlo.

# %%
# [Su código aquí]


# %% [markdown]
# ---
# ## 🏠 Tarea del Día 3: su propia generación en la GPU
#
# En clase usamos estructuras ya generadas. La tarea es enviar **su propio
# trabajo** a la cola de GPU de Kabré (nukwa). No hay que esperar frente a la
# pantalla: el trabajo entra a la cola, corre unos 5 minutos cuando hay una
# GPU libre y el resultado queda guardado en su carpeta.
#
# 1. Abra `tareas/tarea_mattergen.sbatch` y cambie solo las líneas marcadas:
#    `NOMBRE`, `CONDICION` y `VALOR`. Sugerencia: `chemical_system` con
#    `Si-O-N` (cerámicos de alta temperatura) o un sistema químico de su interés.
# 2. Envíelo con la celda de abajo (o en una terminal:
#    `sbatch tareas/tarea_mattergen.sbatch`).
# 3. Cuando termine, cargue sus estructuras y responda:
#    - ¿Cuántas son estructuralmente válidas (distancia mínima > 1.2 Å)?
#    - ¿Cuántas pasan el filtro SMACT? ¿Cuántas son novedosas?
#    - Visualice las 3 mejores y justifique por qué las elegiría para síntesis.

# %%
# Enviar la tarea a la cola nukwa (un solo trabajo de GPU a la vez por estudiante)
sys.path.insert(0, str(CURSO / "tareas"))
from gpu import enviar_a_gpu, estado_gpu
enviar_a_gpu("tareas/tarea_mattergen.sbatch")

# %%
# ¿En qué va? PENDING = esperando GPU, RUNNING = corriendo. Si no aparece, ya terminó.
estado_gpu()

# %%
MI_TAREA = "mi_tarea"          # el NOMBRE que puso en el .sbatch
zip_tarea = SALIDAS / MI_TAREA / "generated_crystals_cif.zip"
if zip_tarea.exists():
    structs_tarea = leer_cifs_zip(zip_tarea)
    print((SALIDAS / MI_TAREA / "resumen.txt").read_text())
else:
    print("Todavía no hay resultados. Revise estado_gpu() y los archivos .out en tareas/registros/")

# %%
# [Su código aquí: filtros, novedad y visualización de sus 3 mejores estructuras]

# %% [markdown]
# ---
# ### 🔜 Conexión con el Día 4
# > Ya sabemos generar candidatos. Pero sintetizar cuesta dinero y tiempo:
# > ¿**cuáles probamos primero** con un presupuesto de 20 experimentos?
# > Mañana: Diseño de Experimentos y Optimización Bayesiana.
