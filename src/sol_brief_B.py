# %% [markdown]
# # ✅ Solución de referencia — Brief B: Recubrimiento PVD
#
# *Notebook del instructor. Una de MUCHAS soluciones válidas.*
#
# **Estrategia elegida:** GP (el modelo correcto para 35 datos) + validación
# leave-one-out + propuesta de 5 condiciones mezclando explotación (EI) y
# exploración dirigida.


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

import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qUpperConfidenceBound
from botorch.optim import optimize_acqf

def encontrar_datos() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents[:3]]:
        if (base / "datos").is_dir():
            return base / "datos"
    raise FileNotFoundError("No encuentro la carpeta datos/")

DATOS = encontrar_datos()
df = pd.read_csv(DATOS / "brief_B_recubrimiento.csv")
print(df.shape)
df.describe().round(2).loc[["min", "mean", "max"]]

# %% [markdown]
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>Sección 1 — EDA: ¿qué exploró la planta?</b> </div>

# %%
proc_cols = ["T_sustrato", "P_N2", "bias_voltaje", "fraccion_Ti", "potencia_kW"]

fig, axes = plt.subplots(1, 5, figsize=(16, 3.2))
for ax, col in zip(axes, proc_cols):
    ax.scatter(df[col], df["dureza_HV"], s=25)
    ax.set_xlabel(col)
axes[0].set_ylabel("dureza_HV")
plt.suptitle("Dureza vs cada variable de proceso (35 experimentos históricos)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4.5))
sns.heatmap(df[proc_cols + ["dureza_HV", "CoF"]].corr(), annot=True,
            cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlaciones del histórico")
plt.tight_layout()
plt.show()

# %% [markdown]
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>Sección 2 — GP + validación leave-one-out</b> </div>
#
# Con 35 puntos, un split 80/20 deja un test de 7 — demasiado ruidoso.
# **Leave-one-out** usa cada punto como test una vez: 35 mini-modelos.

# %%
X = df[proc_cols].values
y = df["dureza_HV"].values
X_min, X_max = X.min(axis=0), X.max(axis=0)
Xn = (X - X_min) / (X_max - X_min)

def gp_fit(Xt, Yt):
    gp = SingleTaskGP(Xt, Yt, outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    return gp

# Leave-one-out
loo_pred, loo_std = [], []
for i in range(len(Xn)):
    mask = np.arange(len(Xn)) != i
    gp = gp_fit(torch.tensor(Xn[mask], dtype=torch.double),
                torch.tensor(y[mask], dtype=torch.double).unsqueeze(-1))
    with torch.no_grad():
        post = gp.posterior(torch.tensor(Xn[[i]], dtype=torch.double))
        loo_pred.append(post.mean.item())
        loo_std.append(post.variance.sqrt().item())

loo_pred, loo_std = np.array(loo_pred), np.array(loo_std)
mae_loo = np.abs(loo_pred - y).mean()
print(f"MAE leave-one-out: {mae_loo:.0f} HV (sobre un rango de {y.max()-y.min():.0f} HV)")

plt.figure(figsize=(6, 6))
plt.errorbar(y, loo_pred, yerr=loo_std, fmt="o", ms=5, alpha=0.7, capsize=3)
lims = [y.min() - 50, y.max() + 50]
plt.plot(lims, lims, "--", color=GRIS, lw=1)
plt.xlabel("Dureza medida (HV)")
plt.ylabel("Dureza predicha LOO (HV)")
plt.title(f"Validación leave-one-out del GP — MAE = {mae_loo:.0f} HV")
plt.tight_layout()
plt.show()

# %% [markdown]
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>Sección 3 — Las próximas 5 condiciones a ensayar</b> </div>
#
# Estrategia: 3 puntos de **explotación** elegidos como LOTE con
# `qUpperConfidenceBound` (q=3, β=2: la adquisición evalúa los 3 puntos juntos,
# así que no gana nada repitiendo el mismo) + 2 de **exploración** (máxima
# incertidumbre del GP) por si el óptimo aparente es un espejismo del modelo.
#
# ¿Por qué no qLogEI? Lo probamos: con q=3 pone un punto en el óptimo y los
# otros dos en zonas de ~2100 HV, porque el GP está tan seguro cerca del mejor
# histórico que casi no hay "mejora esperada" en otro lado. Para un lote de
# explotación, qUCB con β moderado da 3 condiciones distintas y todas duras.

# %%
gp_full = gp_fit(torch.tensor(Xn, dtype=torch.double),
                 torch.tensor(y, dtype=torch.double).unsqueeze(-1))
BOUNDS = torch.stack([torch.zeros(5, dtype=torch.double), torch.ones(5, dtype=torch.double)])
torch.manual_seed(42)

# 3 candidatos de explotación como lote (q=3). Con q=1 repetido tres veces
# el optimizador devolvía tres veces el mismo punto.
qUCB = qUpperConfidenceBound(gp_full, beta=2.0)
lote, _ = optimize_acqf(qUCB, bounds=BOUNDS, q=3, num_restarts=10, raw_samples=256)
elegidos = [lote[[k]] for k in range(3)]
dist = torch.cdist(lote, lote)
print(f"Distancia mínima entre los 3 puntos del lote: {dist[dist > 0].min():.2f} (espacio normalizado)")

# 2 candidatos de exploración: máxima desviación estándar posterior
grid = torch.rand(4000, 5, dtype=torch.double)
with torch.no_grad():
    std_grid = gp_full.posterior(grid).variance.sqrt().squeeze()
for _ in range(2):
    idx = int(std_grid.argmax())
    cand = grid[[idx]]
    elegidos.append(cand)
    lejos = torch.norm(grid - cand, dim=1) > 0.2
    std_grid = torch.where(lejos, std_grid, torch.zeros_like(std_grid))

# Tabla final en unidades reales, con predicción ± incertidumbre
filas = []
for i, cand in enumerate(elegidos):
    with torch.no_grad():
        post = gp_full.posterior(cand)
    real = cand.squeeze().numpy() * (X_max - X_min) + X_min
    filas.append({
        **{c: round(v, 2) for c, v in zip(proc_cols, real)},
        "HV_pred": round(post.mean.item()),
        "±std": round(post.variance.sqrt().item()),
        "tipo": "explotación" if i < 3 else "exploración",
    })
propuesta = pd.DataFrame(filas)
print(propuesta.to_string(index=False))
print(f"\nMejor histórico: {y.max():.0f} HV")

# %% [markdown]
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>Sección 4 — Comunicación</b> </div>

# %%
plt.figure(figsize=(8, 5.5))
plt.scatter(df["T_sustrato"], df["bias_voltaje"], c=df["dureza_HV"],
            cmap="viridis", s=60, edgecolor="k", lw=0.5, label="Histórico (35)")
plt.colorbar(label="dureza_HV")
expl = propuesta[propuesta["tipo"] == "explotación"]
explor = propuesta[propuesta["tipo"] == "exploración"]
plt.scatter(expl["T_sustrato"], expl["bias_voltaje"], marker="*", s=350,
            c=NARANJA, edgecolor="k", label="Propuesta: explotación (3)")
plt.scatter(explor["T_sustrato"], explor["bias_voltaje"], marker="^", s=180,
            c="orange", edgecolor="k", label="Propuesta: exploración (2)")
plt.xlabel("T_sustrato (°C)")
plt.ylabel("bias_voltaje (V)")
plt.title("Histórico de planta y próximos 5 experimentos propuestos")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Párrafo para el jefe de planta (ejemplo):**
#
# > Con los 35 ensayos históricos construimos un modelo que predice la dureza
# > con error típico de ±{MAE} HV. Proponemos 5 ensayos: tres apuntan a la zona
# > que el modelo identifica como la más dura (mejora esperada sobre el mejor
# > histórico), y dos verifican regiones donde la planta nunca ha operado y el
# > modelo tiene máxima incertidumbre. Costo total: 5 turnos (~\$2,500) —
# > frente a los ~20 ensayos que costaría un barrido clásico equivalente.
#
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>Sección 5 — Limitaciones (lo que el equipo DEBE decir)</b> </div>
#
# - 35 puntos en 5 dimensiones es MUY poco: el modelo interpola, la
#   extrapolación fuera del rango histórico no es confiable.
# - El CoF quedó fuera del modelo — antes de fijar condiciones definitivas hay
#   que verificar que las propuestas no lo degraden (objetivo múltiple, o al
#   menos restricción a posteriori).
# - Ruido de medición: las réplicas 9-11 (mismas condiciones) varían ~40 HV.
#   Diferencias menores que eso entre candidatos no son significativas.
# - Deriva de proceso: si la planta cambió de blanco/cámara desde los ensayos
#   históricos, el modelo hereda ese sesgo. Repetir 1-2 puntos históricos como
#   control es barato y lo detecta.
