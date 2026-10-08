# %% [markdown]
# <div style="background:linear-gradient(90deg,#1A2E4A 0%,#2E5E9B 100%);color:white;padding:20px 26px;border-radius:12px;">
# <span style="font-size:12px;letter-spacing:3px;color:#D6E8F7;">CURSO IA APLICADA A CIENCIA DE MATERIALES &nbsp;·&nbsp; DÍA 5 · MINI-RETO</span><br>
# <span style="font-size:26px;font-weight:700;">🛠️ Brief B: Recubrimiento PVD</span><br>
# <span style="font-size:14px;color:#D6E8F7;">35 experimentos históricos, 5 propuestas — 2 h, en equipo</span>
# </div>
#
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>Sección 0 — Contexto del brief (leer, no modificar)</b> </div>
#
# **Cliente:** empresa de herramientas de corte. Deposita (Ti,Al)N por PVD y
# quiere **maximizar la dureza** del recubrimiento manteniendo el coeficiente
# de fricción (CoF) bajo control.
#
# **Lo que tienen:** el histórico completo de la planta — **solo 35
# experimentos** (`brief_B_recubrimiento.csv`). Cada experimento adicional
# cuesta ~\$500 y un turno de 8 horas.
#
# **Su encargo:**
# 1. Analizar el histórico: ¿qué variables de proceso mueven la dureza?
# 2. Construir un modelo (pista de la teoría: con <100 datos, un **GP** es
#    el candidato natural — da incertidumbre, no solo predicción)
# 3. Proponer las **próximas 5 condiciones de proceso a ensayar**, con
#    justificación (¿explotan? ¿exploran? ¿por qué esas 5?)
#
# > 💡 Pista: si optimizan la adquisición 5 veces seguidas con `q=1` van a
# > obtener 5 veces casi el mismo punto. botorch propone lotes de una vez con
# > adquisiciones "q" (p. ej. `qUpperConfidenceBound` o
# > `qLogExpectedImprovement` con `optimize_acqf(..., q=3)`), que tienen en
# > cuenta que los puntos se ensayan juntos. Comparen qué lote propone cada una.
# > Fíjense también en los experimentos repetidos del histórico: ¿qué les
# > dicen sobre el ruido de medición?
#
# **Columnas del dataset:**
# `exp_id`, `T_sustrato` (°C), `P_N2` (Pa), `bias_voltaje` (V),
# `fraccion_Ti`, `potencia_kW`, `dureza_HV`, `CoF`.
#
# **Entregable (5 min de presentación):**
# - Tabla con las 5 condiciones propuestas y su predicción (con incertidumbre)
# - 2–3 figuras que muestren su lógica
# - Limitaciones honestas: ¿qué le advertirían al jefe de planta?
#
# **Rúbrica:** razonamiento del pipeline (40%) · calidad del resultado (25%)
# · comunicación (25%) · limitaciones identificadas (10%)
#
# > 💡 Con 35 datos NO se puede hacer todo lo que hicieron con 1,181. Esa
# > restricción es el corazón del brief — igual que en la planta real.
#
# **Timing sugerido:** EDA 20 min → pipeline 30 min → checkpoint instructor →
# refinamiento 30 min → figuras y conclusión 20 min → ordenar presentación 5 min.

# %% [markdown]
# ---
# ## Herramientas disponibles *(ejecutar tal cual)*


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
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf

def encontrar_datos() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents[:3]]:
        if (base / "datos").is_dir():
            return base / "datos"
    raise FileNotFoundError("No encuentro la carpeta datos/")

DATOS = encontrar_datos()
df = pd.read_csv(DATOS / "brief_B_recubrimiento.csv")
print(f"Histórico de planta: {df.shape[0]} experimentos × {df.shape[1]} columnas")
df.head(3)

# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>Sección 1 — Carga y EDA (su código aquí)</b> </div>
#
# Preguntas guía: ¿qué rango cubrió la planta en cada variable? ¿alguna
# correlación obvia con `dureza_HV`? ¿los 35 puntos cubren bien el espacio o
# hay zonas nunca exploradas?

# %%
# [CELDA LIBRE — su código aquí]


# %%
# [CELDA LIBRE]


# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>Sección 2 — Pipeline de modelado (libre elección)</b> </div>
#
# Guía: ¿GP con qué kernel? ¿escalan las variables primero? ¿cómo validan un
# modelo con solo 35 puntos? (¿leave-one-out? ¿5-fold?) ¿Qué hacen con `CoF`
# — restricción, segundo objetivo, o lo ignoran y lo declaran en limitaciones?
#
# 🎯 Consultar la documentación de sklearn/botorch NO es trampa: en el
# trabajo real, saber buscar ES la habilidad. Lo que no pueden hacer es
# delegar el criterio: cada decisión del pipeline debe tener un porqué.

# %%
# [CELDA LIBRE]


# %%
# [CELDA LIBRE]


# %%
# [CELDA LIBRE]


# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>Sección 3 — Resultado principal (sus 5 condiciones propuestas)</b> </div>
#
# Tabla: T_sustrato, P_N2, bias_voltaje, fraccion_Ti, potencia_kW →
# dureza predicha ± incertidumbre.

# %%
# [CELDA LIBRE — construyan aquí su tabla de 5 condiciones]


# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>Sección 4 — Comunicación del resultado</b> </div>
#
# 2–3 figuras clave + un párrafo de conclusión dirigido **al jefe de planta**
# (le interesan HV, costos y riesgos — no los hiperparámetros del kernel).

# %%
# [CELDA LIBRE — figura 1]


# %%
# [CELDA LIBRE — figura 2]


# %% [markdown]
# *(Escriban aquí su párrafo de conclusión para el jefe de planta)*

# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>Sección 5 — Limitaciones y próximos pasos</b> </div>
#
# ¿Qué harían con más tiempo/datos? ¿Riesgos de extrapolar fuera del rango
# histórico? ¿Qué pasa si la planta cambia de blanco (target) de sputtering?

# %% [markdown]
# *(Su texto aquí)*
