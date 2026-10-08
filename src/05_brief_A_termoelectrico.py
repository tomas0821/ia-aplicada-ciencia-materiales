# %% [markdown]
# <div style="background:linear-gradient(90deg,#1A2E4A 0%,#2E5E9B 100%);color:white;padding:20px 26px;border-radius:12px;">
# <span style="font-size:12px;letter-spacing:3px;color:#D6E8F7;">CURSO IA APLICADA A CIENCIA DE MATERIALES &nbsp;·&nbsp; DÍA 5 · MINI-RETO</span><br>
# <span style="font-size:26px;font-weight:700;">🔥 Brief A: Termoeléctricos</span><br>
# <span style="font-size:14px;color:#D6E8F7;">Recuperación de calor de escape automotriz — 2 h, en equipo</span>
# </div>
#
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>Sección 0 — Contexto del brief (leer, no modificar)</b> </div>
#
# **Cliente:** fabricante de autopartes que quiere recuperar calor del escape
# (400–700 °C) con módulos termoeléctricos.
#
# **Su encargo:** del catálogo de **800 materiales candidatos** del cliente
# (`brief_A_termoelectricos.csv`), proponer una **lista corta de 10
# candidatos** para validación experimental, balanceando:
#
# 1. **Desempeño:** figura de mérito `ZT_estimated` alta
# 2. **Costo:** `costo_relativo` razonable (el cliente es sensible al precio)
# 3. **Confianza:** justificar por qué su modelo/criterio es confiable
#
# **Columnas del dataset:**
# `formula`, `grupo` (familia química), `bandgap_eV`, `seebeck_uV_K`,
# `sigma_S_m` (conductividad eléctrica), `kappa_W_mK` (conductividad térmica),
# `ZT_estimated`, `costo_relativo`, + 6 features Magpie precalculadas.
#
# > ⚠️ `ZT_estimated` = S²σT/κ evaluado alrededor de **T ≈ 600 K (≈ 330 °C)**,
# > con dispersión. El escape trabaja a 400–700 °C (670–970 K) y el ZT depende
# > fuertemente de T: un material con pico de ZT a alta temperatura (p. ej.
# > silicuros, skutteruditas) puede estar subestimado aquí. Mencionarlo cuenta
# > como limitación bien identificada.
#
# **Entregable (5 min de presentación):**
# - Tabla con sus 10 candidatos y el criterio de selección
# - 2–3 figuras clave que defiendan la elección
# - Limitaciones honestas: ¿qué NO saben todavía?
#
# **Rúbrica:** razonamiento del pipeline (40%) · calidad del resultado (25%)
# · comunicación (25%) · limitaciones identificadas (10%)
#
# > 💡 No hay una única respuesta correcta. Un equipo puede filtrar y rankear;
# > otro puede modelar ZT y explorar candidatos fuera del top obvio; otro puede
# > armar un frente de Pareto ZT-costo. **El razonamiento es lo que se evalúa.**
#
# **Timing sugerido:** EDA 20 min → pipeline 30 min → checkpoint instructor →
# refinamiento 30 min → figuras y conclusión 20 min → ordenar presentación 5 min.

# %% [markdown]
# ---
# ## Herramientas disponibles *(ejecutar tal cual)*
#
# Todo lo visto en los Días 1–4 está importado y listo:


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

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import shap

import torch
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

def encontrar_datos() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents[:3]]:
        if (base / "datos").is_dir():
            return base / "datos"
    raise FileNotFoundError("No encuentro la carpeta datos/")

DATOS = encontrar_datos()
df = pd.read_csv(DATOS / "brief_A_termoelectricos.csv")
print(f"Catálogo del cliente: {df.shape[0]} materiales × {df.shape[1]} columnas")
df.head(3)

# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>Sección 1 — Carga y EDA (su código aquí)</b> </div>
#
# Preguntas guía: ¿cómo se distribuye `ZT_estimated`? ¿qué familias (`grupo`)
# dominan el catálogo? ¿hay trade-off visible entre ZT y costo?

# %%
# [CELDA LIBRE — su código aquí]


# %%
# [CELDA LIBRE]


# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>Sección 2 — Pipeline de modelado (libre elección)</b> </div>
#
# Guía: ¿cuál es su target? ¿qué features usan? ¿qué modelo eligen y por qué?
# ¿O deciden que aquí no hace falta un modelo y basta un filtrado inteligente?
# (Ambas rutas son defendibles — expliquen la suya.)
#
# 🎯 Consultar la documentación de sklearn/pandas/botorch NO es trampa:
# en el trabajo real, saber buscar ES la habilidad. Lo que no pueden hacer
# es delegar el criterio: cada decisión del pipeline debe tener un porqué.

# %%
# [CELDA LIBRE]


# %%
# [CELDA LIBRE]


# %%
# [CELDA LIBRE]


# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>Sección 3 — Resultado principal (su tabla de 10 candidatos)</b> </div>

# %%
# [CELDA LIBRE — construyan aquí su tabla final de 10 candidatos]


# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>Sección 4 — Comunicación del resultado</b> </div>
#
# 2–3 figuras clave + un párrafo de conclusión dirigido **al cliente**
# (que no sabe qué es un Random Forest ni le importa).

# %%
# [CELDA LIBRE — figura 1]


# %%
# [CELDA LIBRE — figura 2]


# %% [markdown]
# *(Escriban aquí su párrafo de conclusión para el cliente)*

# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>Sección 5 — Limitaciones y próximos pasos</b> </div>
#
# ¿Qué harían con más tiempo/datos? ¿Qué le advertirían al cliente antes de
# que gaste dinero en síntesis? (Pista de la teoría: el ZT estimado por
# modelos suele sobreestimar el ZT real medido ~2×.)

# %% [markdown]
# *(Su texto aquí)*
