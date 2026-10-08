# %% [markdown]
# # ✅ Solución — Ejercicio del Día 4
#
# *Notebook del instructor.* Campaña PVD con **UCB (β=2.0)** vs **EI**, y
# análisis de la diferencia de exploración.


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
from pathlib import Path
from scipy.interpolate import RBFInterpolator

import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf

def encontrar_datos() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents[:3]]:
        if (base / "datos").is_dir():
            return base / "datos"
    raise FileNotFoundError("No encuentro la carpeta datos/")

DATOS = encontrar_datos()
df_pvd = pd.read_csv(DATOS / "superhard_experimental.csv")

feature_cols_pvd = ["T_proceso", "P_N2", "bias_voltaje", "flujo_Ti_Al"]
X_pvd = df_pvd[feature_cols_pvd].values
y_pvd = df_pvd["dureza_HV"].values

X_min, X_max = X_pvd.min(axis=0), X_pvd.max(axis=0)
X_norm = (X_pvd - X_min) / (X_max - X_min)
oracle = RBFInterpolator(X_norm, y_pvd, kernel="thin_plate_spline")
BOUNDS = torch.stack([torch.zeros(4, dtype=torch.double), torch.ones(4, dtype=torch.double)])

def oracle_torch(X_t):
    return torch.tensor(oracle(X_t.detach().numpy()), dtype=torch.double).unsqueeze(-1)

# %% [markdown]
# ## Campañas EI y UCB con el mismo punto de partida

# %%
def campana(acq_name: str, n_iter: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    mitad_peor = np.argsort(y_pvd)[: len(y_pvd) // 2]
    idx0 = rng.choice(mitad_peor, 5, replace=False)
    tX = torch.tensor(X_norm[idx0], dtype=torch.double)
    tY = torch.tensor(y_pvd[idx0], dtype=torch.double).unsqueeze(-1)
    curva = [tY.max().item()]
    for _ in range(n_iter):
        gp = SingleTaskGP(tX, tY, outcome_transform=Standardize(m=1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        if acq_name == "EI":
            acq = LogExpectedImprovement(gp, best_f=tY.max())
        else:
            acq = UpperConfidenceBound(gp, beta=2.0)
        cand, _ = optimize_acqf(acq, bounds=BOUNDS, q=1, num_restarts=5, raw_samples=32)
        tX = torch.cat([tX, cand])
        tY = torch.cat([tY, oracle_torch(cand)])
        curva.append(tY.max().item())
    return curva, tX, tY

curva_ei, X_ei, Y_ei = campana("EI")
curva_ucb, X_ucb, Y_ucb = campana("UCB")

best_global = y_pvd.max()

def primera_iter(curva, umbral):
    for i, v in enumerate(curva):
        if v >= umbral:
            return i
    return "nunca"

print(f"HV máximo histórico: {best_global:.0f}")
for pct in [95, 98, 99]:
    u = pct / 100 * best_global
    print(f"  {pct}% ({u:.0f} HV): EI en el experimento {primera_iter(curva_ei, u)}, "
          f"UCB en el experimento {primera_iter(curva_ucb, u)}")
print(f"Final: EI {curva_ei[-1]:.0f}  |  UCB {curva_ucb[-1]:.0f}")
# Referencia (prueba local, setiembre 2026): el 95% lo cruzan ambos en el
# experimento 3 y el 98% ambos en el 5; el 99% LogEI en el 6 y UCB en el 16.
# Los números exactos cambian con la versión de botorch; la lección (95% no
# discrimina, 99% sí; UCB con β=2 explora más y tarda más en afinar) se mantiene.

# %%
plt.figure(figsize=(9, 5))
plt.plot(curva_ei, label="LogEI", lw=2, marker="o", ms=4)
plt.plot(curva_ucb, label="UCB (β=2.0)", lw=2, marker="s", ms=4)
plt.axhline(best_global, color="k", ls=":", label="Máximo histórico")
plt.axhline(0.98 * best_global, color="gray", ls=":", lw=1, label="98%")
plt.axhline(0.99 * best_global, color="gray", ls="--", lw=1, label="99%")
plt.xlabel("Experimento")
plt.ylabel("Mejor dureza (HV)")
plt.title("Convergencia EI vs UCB — proceso PVD")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. ¿Cómo explora cada uno? (plano T_proceso vs P_N2)

# %%
def desnormalizar(tX):
    return tX.numpy() * (X_max - X_min) + X_min

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
for ax, (nombre, tX) in zip(axes, [("EI", X_ei), ("UCB β=2.0", X_ucb)]):
    Xr = desnormalizar(tX)
    orden = np.arange(len(Xr))
    sc = ax.scatter(Xr[:, 0], Xr[:, 1], c=orden, cmap="viridis", s=60, edgecolor="k", lw=0.5)
    ax.scatter(Xr[:5, 0], Xr[:5, 1], marker="x", c=NARANJA, s=80, label="5 iniciales")
    ax.set_xlabel("T_proceso (°C)")
    ax.set_title(f"Exploración con {nombre}")
    ax.legend()
axes[0].set_ylabel("P_N2 (Pa)")
fig.colorbar(sc, ax=axes, label="Número de experimento")
plt.show()

# Comentario esperado:
# - EI se vuelve "codicioso" en cuanto encuentra una zona buena: los últimos
#   puntos se apiñan alrededor del óptimo.
# - UCB con β=2.0 mantiene visitas a zonas inexploradas durante más tiempo
#   (paga el bono de incertidumbre), por eso su convergencia final puede ser
#   algo más lenta — pero es más robusto si hubiera óptimos múltiples.
# - Regla práctica: EI como default; UCB si sospechas multimodalidad o si el
#   costo de "quedarse atrapado" es alto.

# %% [markdown]
# ---
# # Retos en clase — Día 4
#
# ## Reto 1: tamaños de diseño para 6 variables

# %%
from pyDOE3 import bbdesign, pbdesign
# (bbdesign y pbdesign no usan azar; en lhs use seed=..., no random_state)

print(f"(a) Factorial completo 2 niveles: 2^6  = {2**6} experimentos")
print(f"(b) Factorial completo 3 niveles: 3^6  = {3**6} experimentos")
print(f"(c) Box-Behnken (center=3):              {bbdesign(6, center=3).shape[0]} experimentos")
print(f"(d) Plackett-Burman:                     {pbdesign(6).shape[0]} experimentos")
# Con presupuesto de 50: Box-Behnken (~52) queda al límite; la respuesta
# defendible es PB (8) para cribar variables + el resto del presupuesto en
# LHS/BO sobre las 3-4 variables que sobreviven. "Gastar todo en un diseño
# rígido" vs "cribar y luego optimizar" es exactamente la discusión buscada.

# %% [markdown]
# ## Reto 2: BO con solo 3 puntos iniciales (Hartmann 6D)

# %%
from botorch.test_functions import Hartmann
from botorch.utils.sampling import draw_sobol_samples

hartmann = Hartmann(dim=6, negate=True)
B6 = torch.stack([torch.zeros(6, dtype=torch.double), torch.ones(6, dtype=torch.double)])

def bo_hartmann(n_init, n_iter, seed):
    torch.manual_seed(seed)
    tX = draw_sobol_samples(bounds=B6, n=n_init, q=1, seed=seed).squeeze(1)
    tY = hartmann(tX).unsqueeze(-1)
    curva = [tY.max().item()]
    for _ in range(n_iter):
        gp = SingleTaskGP(tX, tY, outcome_transform=Standardize(m=1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        acq = LogExpectedImprovement(gp, best_f=tY.max())
        c, _ = optimize_acqf(acq, bounds=B6, q=1, num_restarts=5, raw_samples=64)
        tX = torch.cat([tX, c]); tY = torch.cat([tY, hartmann(c).unsqueeze(-1)])
        curva.append(tY.max().item())
    return curva

OPT = 3.3224
for seed in [1, 2, 3]:
    c3 = bo_hartmann(3, 37, seed)   # 3 + 37 = 40 evaluaciones totales
    print(f"seed {seed}: init=3  → final {c3[-1]:.3f} ({100*c3[-1]/OPT:.0f}% del óptimo)")

# Hallazgo típico: a veces converge igual, a veces se atasca varios
# experimentos en una zona mediocre — la VARIABILIDAD entre semillas crece
# mucho. Recomendación: con pocos puntos iniciales, usar una adquisición más
# exploradora (UCB con β alto) las primeras iteraciones, o aceptar que los
# primeros pasos de BO harán el papel del diseño inicial.

# %% [markdown]
# ## Reto 3: incorporar la restricción CoF < 0.4
# Ruta ilustrada: segundo oráculo para CoF + penalización del objetivo.
# (Ruta avanzada en botorch: ConstrainedExpectedImprovement / qNEHVI.)

# %%
oracle_cof = RBFInterpolator(X_norm, df_pvd["CoF"].values, kernel="thin_plate_spline")

def objetivo_penalizado(X_t, limite=0.4, penal=500.0):
    hv = oracle(X_t.detach().numpy())
    cof = oracle_cof(X_t.detach().numpy())
    return torch.tensor(hv - penal * np.maximum(cof - limite, 0), dtype=torch.double).unsqueeze(-1)

torch.manual_seed(42)
rng = np.random.default_rng(42)
mitad_peor = np.argsort(y_pvd)[: len(y_pvd) // 2]
idx0 = rng.choice(mitad_peor, 5, replace=False)
tX = torch.tensor(X_norm[idx0], dtype=torch.double)
tY = objetivo_penalizado(tX)
for _ in range(20):
    gp = SingleTaskGP(tX, tY, outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    acq = LogExpectedImprovement(gp, best_f=tY.max())
    c, _ = optimize_acqf(acq, bounds=BOUNDS, q=1, num_restarts=5, raw_samples=32)
    tX = torch.cat([tX, c]); tY = torch.cat([tY, objetivo_penalizado(c)])

best = int(tY.argmax())
hv_best = float(oracle(tX[best].unsqueeze(0).numpy())[0])
cof_best = float(oracle_cof(tX[best].unsqueeze(0).numpy())[0])
print(f"Mejor punto factible-penalizado: HV={hv_best:.0f}, CoF={cof_best:.3f} (límite 0.4)")
# Discusión: la penalización es simple y funciona, pero el peso (500) es
# arbitrario — motivación perfecta para mencionar constrained-EI/qNEHVI como
# la versión "de adultos" del mismo concepto.
