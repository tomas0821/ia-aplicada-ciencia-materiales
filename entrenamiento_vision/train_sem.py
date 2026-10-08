"""Ajuste fino de ResNet-18 (ImageNet) para clasificar imágenes SEM de NFFA-Europe en 10 categorías.
Se reservan 20 imágenes por clase como conjunto de prueba para los estudiantes."""
import json, os, random, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from PIL import Image
import torchvision
from torchvision import transforms as T

RAW = Path(os.environ.get("CV_RAW", "/work/trojas/ia_cm_cv/raw"))
OUT = Path(os.environ.get("CV_OUT", "/work/trojas/ia_cm_cv/out")); OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCH_HOME", "/work/trojas/torch_cache")
torch.manual_seed(0); random.seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

clases = sorted(p.name for p in (RAW / "sem").iterdir() if p.is_dir())
archivos = {c: sorted((RAW / "sem" / c).glob("*.png")) for c in clases}
split = {"train": [], "val": [], "test": []}
for k, c in enumerate(clases):
    fs = archivos[c][:]; random.shuffle(fs)
    n_test = 20; n_val = max(10, int(0.12 * len(fs)))
    split["test"] += [(str(f), k) for f in fs[:n_test]]
    split["val"] += [(str(f), k) for f in fs[n_test:n_test + n_val]]
    split["train"] += [(str(f), k) for f in fs[n_test + n_val:]]
print({s: len(v) for s, v in split.items()}, {c: len(archivos[c]) for c in clases}, flush=True)
json.dump({"clases": clases, "split": split}, open(OUT / "sem_split.json", "w"))

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
tr_train = T.Compose([T.Grayscale(3), T.RandomResizedCrop(224, scale=(0.5, 1.0)), T.RandomHorizontalFlip(),
                      T.RandomVerticalFlip(), T.ColorJitter(0.3, 0.3), T.ToTensor(), T.Normalize(MEAN, STD)])
tr_eval = T.Compose([T.Grayscale(3), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])


class DS(torch.utils.data.Dataset):
    def __init__(self, items, tr): self.items, self.tr = items, tr
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        f, y = self.items[i]
        return self.tr(Image.open(f)), y


ytr = np.array([y for _, y in split["train"]])
peso = 1.0 / np.bincount(ytr)[ytr]
sampler = torch.utils.data.WeightedRandomSampler(peso, num_samples=len(ytr), replacement=True)
dl_tr = torch.utils.data.DataLoader(DS(split["train"], tr_train), batch_size=64, sampler=sampler, num_workers=8)
dl_va = torch.utils.data.DataLoader(DS(split["val"], tr_eval), batch_size=128, num_workers=8)
dl_te = torch.utils.data.DataLoader(DS(split["test"], tr_eval), batch_size=128, num_workers=8)

m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
m.fc = nn.Linear(512, len(clases))
m = m.to(dev)
opt = torch.optim.AdamW([{"params": [p for n, p in m.named_parameters() if not n.startswith("fc")], "lr": 2e-4},
                         {"params": m.fc.parameters(), "lr": 2e-3}], weight_decay=1e-4)
EP = int(os.environ.get("EP_SEM", "10"))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2e-4, 2e-3], total_steps=EP * len(dl_tr))
lossf = nn.CrossEntropyLoss(label_smoothing=0.05)


def evaluar(dl):
    m.eval(); ok = n = 0; P = []; Y = []
    with torch.no_grad():
        for x, y in dl:
            p = m(x.to(dev)).argmax(1).cpu(); ok += (p == y).sum().item(); n += len(y); P += p.tolist(); Y += y.tolist()
    return ok / n, P, Y


hist = []
for ep in range(EP):
    m.train(); t = time.time(); tot = 0
    for x, y in dl_tr:
        opt.zero_grad(); l = lossf(m(x.to(dev)), y.to(dev)); l.backward(); opt.step(); sched.step(); tot += l.item()
    acc, _, _ = evaluar(dl_va)
    hist.append({"ep": ep + 1, "loss": tot / len(dl_tr), "val_acc": acc})
    print(hist[-1], f"{time.time()-t:.0f} s", flush=True)

acc_te, P, Y = evaluar(dl_te)
print("test acc", acc_te)
m = m.cpu().half()
torch.save(m.state_dict(), OUT / "sem_resnet18.pt")
json.dump({"clases": clases, "mean": MEAN, "std": STD, "val_hist": hist, "test_acc": acc_te,
           "test_pred": P, "test_true": Y}, open(OUT / "sem_info.json", "w"), indent=1)
print("guardado", (OUT / "sem_resnet18.pt").stat().st_size / 1e6, "MB")
