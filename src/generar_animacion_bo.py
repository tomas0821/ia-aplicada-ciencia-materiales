"""Genera media/bo_animacion.gif — el loop de BO en 1D, frame por frame.

GP (media + banda 2sigma) arriba, Expected Improvement abajo. Cada frame
adquiere el punto que EI recomienda. Paleta validada del curso.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

AZUL, NARANJA, TINTA, GRIS = "#2E5E9B", "#D96F0F", "#1A2E4A", "#8A94A3"

rng = np.random.default_rng(3)

def f(x):
    """Funcion objetivo 'desconocida' con optimo global no trivial."""
    return np.sin(5.5 * x) * (1 - x) + 1.1 * np.exp(-25 * (x - 0.75) ** 2)

X_grid = np.linspace(0, 1, 400)
y_true = f(X_grid)
x_opt = X_grid[np.argmax(y_true)]

# puntos iniciales (deliberadamente lejos del optimo)
X_obs = [0.05, 0.35, 0.55]
y_obs = [float(f(x)) for x in X_obs]

N_FRAMES = 12
estados = []
for _ in range(N_FRAMES):
    gp = GaussianProcessRegressor(
        kernel=1.0 * RBF(length_scale=0.12, length_scale_bounds=(0.05, 0.4))
        + WhiteKernel(1e-6, noise_level_bounds="fixed"),
        normalize_y=True, n_restarts_optimizer=3, random_state=0)
    gp.fit(np.array(X_obs).reshape(-1, 1), y_obs)
    mu, sd = gp.predict(X_grid.reshape(-1, 1), return_std=True)
    best = max(y_obs)
    z = (mu - best) / np.maximum(sd, 1e-9)
    ei = (mu - best) * norm.cdf(z) + sd * norm.pdf(z)
    ei[sd < 1e-9] = 0
    x_next = X_grid[np.argmax(ei)]
    estados.append((list(X_obs), list(y_obs), mu.copy(), sd.copy(), ei.copy(), x_next))
    X_obs.append(float(x_next))
    y_obs.append(float(f(x_next)))

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8, 5.6), sharex=True,
    gridspec_kw={"height_ratios": [2.4, 1]}, dpi=90)
fig.patch.set_facecolor("white")

def dibujar(k):
    Xo, yo, mu, sd, ei, x_next = estados[k]
    for ax in (ax1, ax2):
        ax.clear()
        ax.set_facecolor("white")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(alpha=0.2)
        ax.tick_params(colors=TINTA)
    ax1.plot(X_grid, y_true, ls=":", color=GRIS, lw=1.5, label="función real (oculta)")
    ax1.fill_between(X_grid, mu - 2 * sd, mu + 2 * sd, color=AZUL, alpha=0.18,
                     label="incertidumbre del GP (±2σ)")
    ax1.plot(X_grid, mu, color=AZUL, lw=2.2, label="predicción del GP")
    ax1.scatter(Xo, yo, s=55, color=TINTA, zorder=5, label=f"experimentos ({len(Xo)})")
    ax1.axvline(x_next, color=NARANJA, ls="--", lw=1.5)
    ax1.axvline(x_opt, color=GRIS, ls=":", lw=1)
    ax1.set_ylim(-1.3, 2.1)
    ax1.set_ylabel("respuesta", color=TINTA)
    ax1.set_title(f"Optimización Bayesiana — experimento {k + 1} de {N_FRAMES}",
                  color=TINTA, fontweight="bold")
    ax1.legend(loc="lower left", fontsize=8, frameon=False, ncol=2)
    ax2.fill_between(X_grid, 0, ei, color=NARANJA, alpha=0.35)
    ax2.plot(X_grid, ei, color=NARANJA, lw=2, label="Expected Improvement")
    ax2.axvline(x_next, color=NARANJA, ls="--", lw=1.5, label="siguiente experimento")
    ax2.set_xlabel("variable de proceso (normalizada)", color=TINTA)
    ax2.set_ylabel("EI", color=TINTA)
    ax2.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()

anim = FuncAnimation(fig, dibujar, frames=N_FRAMES, interval=900)
out = Path(__file__).resolve().parent.parent / "notebooks" / "media" / "bo_animacion.gif"
out.parent.mkdir(parents=True, exist_ok=True)
anim.save(str(out), writer=PillowWriter(fps=1.1))
print(f"GIF: {out} ({out.stat().st_size/1e6:.1f} MB, {N_FRAMES} frames)")
