# %% [markdown]
# <div style="background:linear-gradient(90deg,#1A2E4A 0%,#2E5E9B 100%);color:white;padding:20px 26px;border-radius:12px;">
# <span style="font-size:12px;letter-spacing:3px;color:#D6E8F7;">CURSO IA APLICADA A CIENCIA DE MATERIALES &nbsp;·&nbsp; PRE-CURSO</span><br>
# <span style="font-size:26px;font-weight:700;">✅ Test del Entorno</span><br>
# <span style="font-size:14px;color:#D6E8F7;">Ejecuta esto ANTES del Día 1 — debe terminar en 🎉</span>
# </div>
#
# **Ejecuta este notebook ANTES del Día 1 del curso.**
#
# Su único propósito es confirmar que tu instalación funciona. Ejecuta las celdas
# en orden (`Shift + Enter`). Al final deberías ver el mensaje:
#
# > 🎉 **ENTORNO LISTO** — nos vemos en el Día 1.
#
# Si alguna celda falla, copia el mensaje de error completo y publícalo en el
# canal de soporte del curso, junto con tu sistema operativo.
#
# > **Tiempo estimado:** 2–3 minutos.
# >
# > **En Kabré:** abre este notebook desde la carpeta `curso/notebooks/` y
# > elige el kernel **Python-AI-Materials** (arriba a la derecha).

# %% [markdown]
# ## 1. Versión de Python
#
# El curso requiere **Python ≥ 3.10** (la versión del kernel
# Python-AI-Materials de Kabré, compatible con MatterGen).

# %%
import sys

print(f"Python {sys.version}")
assert sys.version_info >= (3, 10), (
    "❌ Necesitas Python >= 3.10. Revisa que el kernel seleccionado "
    "sea Python-AI-Materials (arriba a la derecha del notebook)."
)
print("✓ Versión de Python correcta")

# %% [markdown]
# ## 2. Paquetes instalados
#
# La celda siguiente intenta importar cada paquete del curso y muestra una tabla
# con el resultado. **Todos deben aparecer con ✓.**

# %%
from importlib import import_module
from importlib.metadata import version as pkg_version

PAQUETES = [
    # (nombre para import, nombre en PyPI, día en que se usa)
    ("numpy",        "numpy",         "todos"),
    ("pandas",       "pandas",        "todos"),
    ("matplotlib",   "matplotlib",    "todos"),
    ("seaborn",      "seaborn",       "todos"),
    ("pymatgen",     "pymatgen",      "D1-D3"),
    ("ase",          "ase",           "D1"),
    ("matminer",     "matminer",      "D1, D2, D5"),
    ("sklearn",      "scikit-learn",  "D2, D5"),
    ("shap",         "shap",          "D2"),
    ("mp_api",       "mp-api",        "D2"),
    ("dotenv",       "python-dotenv", "D2"),
    ("torch",        "torch",         "D3, D4"),
    ("gpytorch",     "gpytorch",      "D4"),
    ("botorch",      "botorch",       "D4"),
    ("pyDOE3",       "pyDOE3",        "D4"),
    ("smact",        "smact",         "D3"),
]

fallos = []
print(f"{'paquete':<15} {'versión':<12} {'días':<12} estado")
print("-" * 50)
for mod, pypi, dias in PAQUETES:
    try:
        import_module(mod)
        ver = pkg_version(pypi)
        print(f"{pypi:<15} {ver:<12} {dias:<12} ✓")
    except Exception as e:
        fallos.append((pypi, str(e)))
        print(f"{pypi:<15} {'---':<12} {dias:<12} ❌ FALTA")

if fallos:
    print("\n❌ Faltan paquetes. Revisa que el kernel sea Python-AI-Materials")
    print("   y, si el problema sigue, avisa en el canal de soporte.")
else:
    print("\n✓ Todos los paquetes presentes")

# %% [markdown]
# ## 3. Test funcional rápido
#
# Importar no basta: ahora comprobamos que cada pieza clave **hace algo**.
# Cada bloque imprime una línea si funciona.

# %%
# --- scikit-learn: entrenar un modelo diminuto ---
import numpy as np
from sklearn.ensemble import RandomForestRegressor

X = np.random.rand(50, 3)
y = X.sum(axis=1)
rf = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
assert rf.predict(X[:1]).shape == (1,)
print("✓ scikit-learn entrena y predice")

# %%
# --- pymatgen: crear una estructura cristalina ---
from pymatgen.core import Lattice, Structure

nacl = Structure.from_spacegroup(
    "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
)
assert len(nacl) == 8
print(f"✓ pymatgen crea estructuras (NaCl: {len(nacl)} sitios)")

# %%
# --- matminer: calcular descriptores Magpie de una composición ---
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

ep = ElementProperty.from_preset("magpie")
feats = ep.featurize(Composition("Fe2O3"))
assert len(feats) == 132
print(f"✓ matminer calcula descriptores Magpie ({len(feats)} features)")

# %%
# --- torch + botorch: ajustar un GP de juguete ---
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

train_X = torch.rand(10, 2, dtype=torch.double)
train_Y = train_X.sum(dim=1, keepdim=True)
gp = SingleTaskGP(train_X, train_Y)
fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
print("✓ botorch ajusta un Gaussian Process")

# %%
# --- pyDOE3: generar un Latin Hypercube ---
from pyDOE3 import lhs

muestra = lhs(4, samples=20, random_state=42)
assert muestra.shape == (20, 4)
print("✓ pyDOE3 genera diseños de experimentos")

# %%
# --- matplotlib: generar una figura ---
import matplotlib

matplotlib.use("Agg")  # no necesita pantalla para el test
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4, 3))
ax.plot([0, 1, 2], [0, 1, 4], "o-")
ax.set_title("Test de matplotlib")
plt.close(fig)
print("✓ matplotlib genera figuras")

# %% [markdown]
# ## 4. Descarga de datasets (matminer)
#
# El Día 1 usamos el dataset `elastic_tensor_2015`. El curso lo trae ya
# descargado en `datos/matminer_cache/`; la variable `MATMINER_DATA` le dice
# a matminer que lo lea de ahí (en Kabré el entorno es de solo lectura y no
# puede descargarlo por su cuenta).

# %%
import os
from pathlib import Path

def encontrar_datos() -> Path:
    """Busca la carpeta datos/ en el directorio actual o sus padres."""
    for base in [Path.cwd(), *Path.cwd().parents[:3]]:
        candidata = base / "datos"
        if candidata.is_dir():
            return candidata
    raise FileNotFoundError("No encuentro la carpeta datos/: abre el notebook desde curso/notebooks/")

DATOS = encontrar_datos()
os.environ["MATMINER_DATA"] = str(DATOS / "matminer_cache")

from matminer.datasets import load_dataset

df = load_dataset("elastic_tensor_2015")
assert df.shape[0] == 1181
print(f"✓ Dataset elastic_tensor_2015 en caché: {df.shape[0]} materiales, {df.shape[1]} columnas")

# %% [markdown]
# ## 5. GPU (solo relevante para el Día 3)
#
# El Día 3 (modelos generativos) corre en los **nodos Nukwa (GPU) de Kabré**.
# Los demás días usan nodos sin GPU. Esta celda solo informa; **no es un error** si dice "CPU solamente".

# %%
import torch

if torch.cuda.is_available():
    print(f"✓ GPU disponible: {torch.cuda.get_device_name(0)}")
else:
    print("ℹ CPU solamente — perfecto para los Días 1, 2, 4 y 5.")
    print("  (El Día 3 usaremos los nodos Nukwa de Kabré, con GPU)")

# %% [markdown]
# ## Resultado final

# %%
if not fallos:
    print("🎉 ENTORNO LISTO — nos vemos en el Día 1.")
else:
    print("❌ Hay paquetes faltantes — revisa la sección 2 y pide ayuda en el canal de soporte.")
