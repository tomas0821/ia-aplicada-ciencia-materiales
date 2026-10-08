"""Tarea de visión computacional: entrenamiento corto de la CNN 1D de XRD.

Pregunta del experimento: ¿qué parte del aumento de datos ayuda a que un modelo
entrenado solo con XRD simulados funcione con difractogramas reales?
Se ejecuta desde tarea_vision.sbatch. Escribe en salidas_vision/<nombre>/:
  xrd_cnn1d.pt  (pesos, mismo formato que el modelo del curso)
  info.json     (opciones, curva de validación y exactitudes)
"""
import argparse, json, math, os, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

CURSO = Path(__file__).resolve().parents[1]
VISION = CURSO / "datos" / "vision"
p = argparse.ArgumentParser()
p.add_argument("--nombre", required=True)
p.add_argument("--epocas", type=int, default=20)
p.add_argument("--aumento", default="completo",
               choices=["completo", "ninguno", "sin_fondo", "sin_kalfa2", "sin_ruido", "sin_ensanchamiento"])
p.add_argument("--fraccion", type=float, default=1.0, help="fracción del conjunto de entrenamiento (0-1)")
p.add_argument("--semilla", type=int, default=0)
a = p.parse_args()
OUT = CURSO / "salidas_vision" / a.nombre
OUT.mkdir(parents=True, exist_ok=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("Dispositivo:", torch.cuda.get_device_name(0) if dev == "cuda" else "CPU (será lento)", flush=True)
torch.manual_seed(a.semilla); np.random.seed(a.semilla)
X0, X1, NX = 10.0, 70.0, 1200
GRID = torch.linspace(X0, X1, NX, device=dev)
AUG = {k: a.aumento != "ninguno" and a.aumento != f"sin_{k}" for k in ["fondo", "kalfa2", "ruido", "ensanchamiento"]}
print("Aumento de datos:", AUG, flush=True)

d = np.load(VISION / "xrd_picos.npz", allow_pickle=True)
SIST = [str(s) for s in d["sistemas"]]
pos, inte, y, formula = d["pos"], d["inte"], d["y"], d["formula"]
# misma partición que el modelo del curso: por fórmula, semilla 0
uf = np.unique(formula); rng = np.random.default_rng(0); rng.shuffle(uf)
n = len(uf); te_f = set(uf[: int(0.10 * n)]); va_f = set(uf[int(0.10 * n): int(0.15 * n)])
part = np.array([2 if f in te_f else (1 if f in va_f else 0) for f in formula])
tr, va, te = (np.where(part == k)[0] for k in range(3))
if a.fraccion < 1:
    tr = np.random.default_rng(a.semilla).permutation(tr)[: int(a.fraccion * len(tr))]
print(f"entrenamiento {len(tr)} | validación {len(va)} | prueba {len(te)}", flush=True)
P = torch.tensor(pos, device=dev); I = torch.tensor(inte, device=dev); Y = torch.tensor(y, device=dev)


def perfiles(idx, aug=None, g=None):
    """Listas de picos -> perfiles de 1200 puntos. aug=None usa las opciones AUG."""
    aug = AUG if aug is None else aug
    p, it = P[idx], I[idx].clone()
    B = len(idx)
    r = lambda *s: torch.rand(*s, device=dev, generator=g)
    shift = torch.zeros(B, 1, device=dev); w = torch.full((B, 1), 0.2, device=dev); eta = torch.full((B, 1), 0.5, device=dev)
    if aug["ensanchamiento"]:
        shift = (r(B, 1) - 0.5) * 0.3
        w = 0.08 + r(B, 1) * 0.42
        eta = r(B, 1)
    if aug["ruido"]:
        it = it * torch.exp(torch.randn(it.shape, device=dev, generator=g) * 0.25)
        it = torch.where((it < 5) & (r(it.shape) < 0.3), torch.zeros_like(it), it)
    if aug["kalfa2"]:
        th = torch.deg2rad(p / 2)
        p2 = 2 * torch.rad2deg(torch.asin((torch.sin(th) * 1.54439 / 1.54056).clamp(max=1)))
        usa = (r(B, 1) < 0.7).float()
        p = torch.cat([p, torch.where(p > 0, p2, torch.zeros_like(p2))], 1)
        it = torch.cat([it, it * 0.5 * usa], 1)
    c = (p + shift)[:, :, None]
    dx2 = (GRID[None, None, :] - c) ** 2
    ww = w[:, :, None] ** 2
    prof = eta[:, :, None] / (1 + 4 * dx2 / ww) + (1 - eta[:, :, None]) * torch.exp(-4 * math.log(2) * dx2 / ww)
    prof = (prof * (it * (p > 0))[:, :, None]).sum(1)
    prof = prof / prof.amax(1, keepdim=True).clamp_min(1e-6)
    if aug["fondo"]:
        x = GRID[None, :]
        bg = r(B, 1) * 0.08 + r(B, 1) * 0.2 * torch.exp(-(x - X0) / (2 + 10 * r(B, 1)))
        bg = bg + r(B, 1) * 0.15 * torch.exp(-((x - (15 + 20 * r(B, 1))) ** 2) / (2 * (3 + 7 * r(B, 1)) ** 2))
        prof = prof + bg
    if aug["ruido"]:
        prof = prof + torch.randn(prof.shape, device=dev, generator=g) * (r(B, 1) * 0.04) \
            + torch.randn(prof.shape, device=dev, generator=g) * torch.sqrt(prof.clamp_min(0)) * (r(B, 1) * 0.04)
    prof = prof.clamp_min(0)
    return prof / prof.amax(1, keepdim=True).clamp_min(1e-6)


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


m = CNN1D(len(SIST)).to(dev)
frec = np.bincount(y[tr], minlength=7).astype(float)
peso = torch.tensor((frec.sum() / (7 * np.maximum(frec, 1))) ** 0.5, dtype=torch.float32, device=dev)
lossf = nn.CrossEntropyLoss(weight=peso)
BS = 256
opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.epocas * math.ceil(len(tr) / BS))
TODO = {k: True for k in AUG}      # validación y prueba siempre con datos "realistas"


def exactitud(idx):
    m.eval(); ok = 0
    g = torch.Generator(device=dev); g.manual_seed(123)
    with torch.no_grad():
        for k in range(0, len(idx), 1024):
            b = torch.tensor(idx[k:k + 1024], device=dev)
            ok += (m(perfiles(b, TODO, g)).argmax(1) == Y[b]).sum().item()
    return ok / len(idx)


t0 = time.time(); hist = []
for ep in range(a.epocas):
    m.train(); perm = np.random.permutation(tr); tot = 0
    for k in range(0, len(perm), BS):
        b = torch.tensor(perm[k:k + BS], device=dev)
        opt.zero_grad(); l = lossf(m(perfiles(b)), Y[b]); l.backward(); opt.step(); sched.step(); tot += l.item()
    hist.append({"epoca": ep + 1, "perdida": tot / math.ceil(len(tr) / BS), "val": exactitud(va)})
    print(hist[-1], f"{time.time() - t0:.0f} s", flush=True)

info = {"opciones": vars(a), "aumento": AUG, "historial": hist, "segundos": round(time.time() - t0),
        "exactitud_simulados": exactitud(te)}
m = m.cpu().eval()
torch.save(m.state_dict(), OUT / "xrd_cnn1d.pt")
real = np.load(VISION / "xrd_reales_opxrd.npz", allow_pickle=True)
Xr = real["perfiles"].astype(np.float32)
if Xr.shape[1] != NX:
    xg = np.linspace(X0, X1, NX)
    Xr = np.stack([np.interp(xg, real["x"], f) for f in Xr]).astype(np.float32)
with torch.no_grad():
    info["exactitud_reales"] = float((m(torch.tensor(Xr)).argmax(1).numpy() == real["y"]).mean())
info["n_reales"] = int(len(Xr))
json.dump(info, open(OUT / "info.json", "w"), indent=1, ensure_ascii=False)
print(f"\nExactitud en simulados de prueba: {info['exactitud_simulados']:.1%}")
print(f"Exactitud en {len(Xr)} XRD reales (opXRD): {info['exactitud_reales']:.1%}")
print(f"Tiempo total: {info['segundos']} s")
