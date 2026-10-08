# %% [markdown]
# # ✅ Solución — Retos de visión computacional (Día 3, P3.5)
#
# *Notebook del instructor.* Reto A (robustez del modelo XRD), Reto B (imágenes
# SEM con menor confianza) y un ejemplo del ejercicio para casa.

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
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import torchvision
from torchvision import transforms as T


def encontrar_datos() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents[:3]]:
        if (base / "datos").is_dir():
            return base / "datos"
    raise FileNotFoundError("No encuentro la carpeta datos/")


VISION = encontrar_datos() / "vision"
torch.set_grad_enabled(False)


class CNN1D(nn.Module):
    def __init__(self, n_clases=7):
        super().__init__()
        capas, cin = [], 1
        for cout, k in [(32, 15), (64, 9), (128, 7), (128, 5), (256, 3)]:
            capas += [nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.ReLU(), nn.MaxPool1d(2)]
            cin = cout
        self.features = nn.Sequential(*capas)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * cin, 128), nn.ReLU(), nn.Linear(128, n_clases))

    def forward(self, x):
        h = self.features(torch.sqrt(x.clamp_min(0))[:, None, :])
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)
        return self.head(h)


sim = np.load(VISION / "xrd_simulados_prueba.npz", allow_pickle=True)
x2t, X_sim, y_sim = sim["x"], sim["perfiles"].astype(np.float32), sim["y"]
NOMBRES = ["triclínico", "monoclínico", "ortorrómbico", "tetragonal", "trigonal", "hexagonal", "cúbico"]
modelo_xrd = CNN1D(7)
modelo_xrd.load_state_dict(torch.load(VISION / "modelos" / "xrd_cnn1d.pt", map_location="cpu"))
modelo_xrd.eval()


def predecir_xrd(X):
    return torch.softmax(modelo_xrd(torch.tensor(X)), 1).numpy()


def perturbar(X, desplazamiento=0.0, ruido=0.0, semilla=0):
    rng = np.random.default_rng(semilla)
    paso = x2t[1] - x2t[0]
    n = int(round(desplazamiento / paso))
    Xp = np.roll(X, n, axis=1)
    if n > 0:
        Xp[:, :n] = X[:, :1]
    elif n < 0:
        Xp[:, n:] = X[:, -1:]
    Xp = np.clip(Xp + rng.normal(0, ruido, X.shape), 0, None)
    return (Xp / Xp.max(1, keepdims=True)).astype(np.float32)

# %% [markdown]
# ## Reto A: robustez frente a desplazamiento y ruido

# %%
desps = np.arange(0, 1.01, 0.1)
ruidos = [0.0, 0.03, 0.08]
fig, ax = plt.subplots(figsize=(8, 4.5))
for r in ruidos:
    acc = [(predecir_xrd(perturbar(X_sim, d, r)).argmax(1) == y_sim).mean() for d in desps]
    ax.plot(desps, acc, marker="o", ms=4, label=f"ruido σ = {r}")
ax.axhline(1 / 7, color=GRIS, ls=":", label="azar")
ax.set_xlabel("desplazamiento en 2θ (grados)"); ax.set_ylabel("exactitud")
ax.set_title("Robustez del modelo XRD"); ax.legend(); plt.tight_layout(); plt.show()

# Lectura esperada: el modelo tolera corrimientos pequeños (se entrenó con
# ±0.15°) y la exactitud cae a medida que el corrimiento supera ese rango. El
# ruido moderado afecta menos que el corrimiento. En la práctica: calibrar el
# cero del difractómetro importa tanto como el modelo.

# %% [markdown]
# ## Reto B: las imágenes SEM con menor confianza

# %%
info = json.load(open(VISION / "modelos" / "sem_info.json"))
CLASES = info["clases"]
prep = T.Compose([T.Grayscale(3), T.CenterCrop(224), T.ToTensor(), T.Normalize(info["mean"], info["std"])])
archivos, y_sem = [], []
for k, c in enumerate(CLASES):
    fs = sorted((VISION / "sem_prueba" / c).glob("*.png")); archivos += fs; y_sem += [k] * len(fs)
y_sem = np.array(y_sem)
m = torchvision.models.resnet18(weights=None); m.fc = nn.Linear(512, len(CLASES))
m.load_state_dict({k: v.float() for k, v in torch.load(VISION / "modelos" / "sem_resnet18.pt", map_location="cpu").items()})
m.eval()
X_img = torch.stack([prep(Image.open(f)) for f in archivos])
prob = torch.cat([torch.softmax(m(X_img[k:k + 50]), 1) for k in range(0, len(X_img), 50)]).numpy()

conf = prob.max(1)
peores = np.argsort(conf)[:5]
fig, axes = plt.subplots(1, 5, figsize=(15, 3.6))
for ax, i in zip(axes, peores):
    top2 = np.argsort(prob[i])[::-1][:2]
    ax.imshow(Image.open(archivos[i]), cmap="gray"); ax.axis("off")
    ax.set_title(f"real: {CLASES[y_sem[i]]}\n{CLASES[top2[0]]} {prob[i, top2[0]]:.0%} · {CLASES[top2[1]]} {prob[i, top2[1]]:.0%}", fontsize=8)
plt.tight_layout(); plt.show()

for u in [0.5, 0.7, 0.9]:
    ok = conf >= u
    print(f"umbral {u:.1f}: decide solo en {ok.mean():.0%} de las imágenes, con exactitud {(prob[ok].argmax(1) == y_sem[ok]).mean():.1%}")

# Discusión: las de menor confianza suelen mezclar dos categorías reales
# (partículas sobre una superficie con patrón, polvo que parece partículas).
# Un umbral convierte al modelo en un asistente: decide lo fácil y deja lo
# dudoso a una persona. Más umbral = más exactitud pero menos cobertura.

# %% [markdown]
# ## Ejercicio para casa: un difractograma propio
# Ejemplo con un patrón experimental de opXRD (como si fuera un archivo xy).

# %%
real = np.load(VISION / "xrd_reales_opxrd.npz", allow_pickle=True)
i = 0
tt, it = real["x"], real["perfiles"][i].astype(float)       # en su caso: tt, it = np.loadtxt("mi_patron.xy", unpack=True)
y = np.interp(x2t, tt, it); y = y - y.min(); y = (y / y.max()).astype(np.float32)
p = predecir_xrd(y[None])[0]
for k in np.argsort(p)[::-1][:3]:
    print(f"{NOMBRES[k]:14s} {p[k]:.1%}")
print("real:", NOMBRES[real["y"][i]], "·", real["composicion"][i])
