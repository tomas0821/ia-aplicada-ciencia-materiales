"""Entrena una CNN 1D que clasifica el sistema cristalino a partir de un patrón XRD (Cu Ka, 2θ 10–70°).
Los perfiles se generan al vuelo desde las listas de picos, con ensanchamiento, desplazamiento,
fondo y ruido aleatorios (aumento de datos), para que el modelo resista mejor los datos reales."""
import json, math, os, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

RAW = Path(os.environ.get("CV_RAW", "/work/trojas/ia_cm_cv/raw"))
OUT = Path(os.environ.get("CV_OUT", "/work/trojas/ia_cm_cv/out")); OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
X0, X1, NX = 10.0, 70.0, 1200
GRID = torch.linspace(X0, X1, NX, device=dev)

d = np.load(RAW / "xrd_picos.npz", allow_pickle=True)
SIST = [str(s) for s in d["sistemas"]]
pos, inte, y, formula = d["pos"], d["inte"], d["y"], d["formula"]
print("patrones", len(y), np.bincount(y, minlength=7), flush=True)

# partición por fórmula: los polimorfos de una fórmula quedan del mismo lado
uf = np.unique(formula); rng = np.random.default_rng(0); rng.shuffle(uf)
n = len(uf); te_f = set(uf[: int(0.10 * n)]); va_f = set(uf[int(0.10 * n): int(0.15 * n)])
part = np.array([2 if f in te_f else (1 if f in va_f else 0) for f in formula])
tr, va, te = (np.where(part == k)[0] for k in range(3))
print("train/val/test", len(tr), len(va), len(te), flush=True)
P = torch.tensor(pos, device=dev); I = torch.tensor(inte, device=dev); Y = torch.tensor(y, device=dev)


def perfiles(idx, aug=True, g=None):
    """Convierte listas de picos en perfiles de 1200 puntos, normalizados a máximo 1."""
    p, it = P[idx], I[idx].clone()
    B = len(idx)
    r = lambda *s: torch.rand(*s, device=dev, generator=g)
    if aug:
        shift = (r(B, 1) - 0.5) * 0.3
        w = 0.08 + r(B, 1) * 0.42
        eta = r(B, 1)
        it = it * torch.exp(torch.randn(it.shape, device=dev, generator=g) * 0.25)
        it = torch.where((it < 5) & (r(it.shape) < 0.3), torch.zeros_like(it), it)
    else:
        shift = torch.zeros(B, 1, device=dev); w = torch.full((B, 1), 0.2, device=dev); eta = torch.full((B, 1), 0.5, device=dev)
    if aug:
        # doblete Cu Kα1/Kα2 en el 70 % de los patrones (los difractómetros de laboratorio lo muestran)
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
    if aug:
        x = GRID[None, :]
        bg = r(B, 1) * 0.08 + r(B, 1) * 0.2 * torch.exp(-(x - X0) / (2 + 10 * r(B, 1)))
        bg = bg + r(B, 1) * 0.15 * torch.exp(-((x - (15 + 20 * r(B, 1))) ** 2) / (2 * (3 + 7 * r(B, 1)) ** 2))
        prof = prof + bg
        prof = prof + torch.randn(prof.shape, device=dev, generator=g) * (r(B, 1) * 0.04) \
            + torch.randn(prof.shape, device=dev, generator=g) * torch.sqrt(prof.clamp_min(0)) * (r(B, 1) * 0.04)
        prof = prof.clamp_min(0)
        prof = prof / prof.amax(1, keepdim=True).clamp_min(1e-6)
    return prof


class CNN1D(nn.Module):
    """Cinco bloques convolución + normalización + ReLU + submuestreo, y dos capas finales."""
    def __init__(self, n_clases=7):
        super().__init__()
        capas, cin = [], 1
        for cout, k in [(32, 15), (64, 9), (128, 7), (128, 5), (256, 3)]:
            capas += [nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.ReLU(), nn.MaxPool1d(2)]
            cin = cout
        self.features = nn.Sequential(*capas)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * cin, 128), nn.ReLU(), nn.Linear(128, n_clases))

    def forward(self, x):                       # x: (lote, 1200)
        h = self.features(torch.sqrt(x.clamp_min(0))[:, None, :])   # raíz: realza picos débiles
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)
        return self.head(h)


m = CNN1D(len(SIST)).to(dev)
frec = np.bincount(y[tr], minlength=7).astype(float)
peso = torch.tensor((frec.sum() / (7 * frec)) ** 0.5, dtype=torch.float32, device=dev)
lossf = nn.CrossEntropyLoss(weight=peso)
EP = int(os.environ.get("EP_XRD", "150")); BS = 256
opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=EP * math.ceil(len(tr) / BS))


def evaluar(idx, aug):
    m.eval(); pred = []
    g = torch.Generator(device=dev); g.manual_seed(123)
    with torch.no_grad():
        for k in range(0, len(idx), 1024):
            b = torch.tensor(idx[k:k + 1024], device=dev)
            pred.append(m(perfiles(b, aug, g)).argmax(1).cpu())
    pred = torch.cat(pred).numpy()
    return (pred == y[idx]).mean(), pred


hist = []
for ep in range(EP):
    m.train(); t = time.time(); perm = np.random.permutation(tr); tot = 0
    for k in range(0, len(perm), BS):
        b = torch.tensor(perm[k:k + BS], device=dev)
        opt.zero_grad(); l = lossf(m(perfiles(b)), Y[b]); l.backward(); opt.step(); sched.step(); tot += l.item()
    acc_va, _ = evaluar(va, True)
    hist.append({"ep": ep + 1, "loss": tot / math.ceil(len(tr) / BS), "val_acc": float(acc_va)})
    print(hist[-1], f"{time.time()-t:.1f} s", flush=True)

acc_te_aug, pred_aug = evaluar(te, True)
acc_te_lim, pred_lim = evaluar(te, False)
print("test sim (con ruido)", acc_te_aug, " test sim (limpio)", acc_te_lim)
m = m.cpu()
torch.save(m.state_dict(), OUT / "xrd_cnn1d.pt")

# conjunto de prueba simulado para los estudiantes: hasta 100 por clase, con ruido fijo
rng = np.random.default_rng(1)
sel = np.concatenate([rng.permutation(te[y[te] == k])[:100] for k in range(7)])
g = torch.Generator(device=dev); g.manual_seed(7)
with torch.no_grad():
    prof = perfiles(torch.tensor(sel, device=dev), True, g).cpu().numpy().astype(np.float16)
np.savez_compressed(OUT / "xrd_simulados_prueba.npz", x=np.linspace(X0, X1, NX).astype(np.float32), perfiles=prof,
                    y=y[sel], formula=formula[sel], sg=d["sg"][sel], sistemas=np.array(SIST))
info = {"sistemas": SIST, "x0": X0, "x1": X1, "nx": NX, "hist": hist, "acc_test_ruido": float(acc_te_aug),
        "acc_test_limpio": float(acc_te_lim), "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te))}

# datos reales (opXRD), si ya se prepararon
real = RAW / "xrd_reales.npz"
if real.exists():
    r = np.load(real, allow_pickle=True)
    mm = CNN1D(len(SIST)); mm.load_state_dict(torch.load(OUT / "xrd_cnn1d.pt")); mm.eval()
    with torch.no_grad():
        pr = mm(torch.tensor(r["perfiles"], dtype=torch.float32)).argmax(1).numpy()
    info["acc_reales"] = float((pr == r["y"]).mean()); info["n_reales"] = int(len(pr))
    print("reales opXRD", info["acc_reales"], len(pr))
json.dump(info, open(OUT / "xrd_info.json", "w"), indent=1)
print("listo")
