"""Extrae de opXRD los patrones experimentales con una sola fase etiquetada (sistema cristalino),
medidos con Cu (λ ≈ 1.54 Å) y que cubren 2θ 10–70°. Los interpola a la misma rejilla del modelo."""
import json, zipfile, os, collections
from pathlib import Path
import numpy as np

RAW = Path(os.environ.get("CV_RAW", "/work/trojas/ia_cm_cv/raw"))
SIST = ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"]
X = np.linspace(10.0, 70.0, 1200)
z = zipfile.ZipFile(RAW / "opxrd.zip")
fuera = collections.Counter(); filas = []
for n in z.namelist():
    if not n.endswith(".json"):
        continue
    d = json.loads(z.read(n))
    lab = d.get("label")
    lab = json.loads(lab) if isinstance(lab, str) else (lab or {})
    fases = [json.loads(p) if isinstance(p, str) else p for p in lab.get("phases", [])]
    fases = [p for p in fases if p and p.get("crystal_system") not in (None, "nan")]
    if len(fases) != 1:
        fuera["sin fase única etiquetada"] += 1; continue
    f = fases[0]
    cs = str(f.get("crystal_system")).lower()
    if cs == "rhombohedral":
        cs = "trigonal"
    if cs not in SIST:
        fuera["sistema desconocido " + cs] += 1; continue
    xi = lab.get("xray_info") or {}
    xi = json.loads(xi) if isinstance(xi, str) else xi
    wl = xi.get("primary_wavelength")
    try:
        wl = float(wl)
    except Exception:
        wl = None
    if wl is None or abs(wl - 1.54) > 0.02:
        fuera["no es Cu Kα"] += 1; continue
    tt = np.asarray(d["two_theta_values"], float); it = np.asarray(d["intensities"], float)
    ok = np.isfinite(tt) & np.isfinite(it)
    tt, it = tt[ok], it[ok]
    if len(tt) < 200 or tt.min() > 10.5 or tt.max() < 69.5:
        fuera["no cubre 10–70°"] += 1; continue
    o = np.argsort(tt)
    tt, it = tt[o], it[o]
    # promedio por intervalos de 0.05° (reduce el ruido de pasos finos) y luego interpolación
    borde = np.concatenate([[X[0] - 0.025], (X[1:] + X[:-1]) / 2, [X[-1] + 0.025]])
    k = np.digitize(tt, borde) - 1
    dentro = (k >= 0) & (k < len(X))
    suma = np.bincount(k[dentro], weights=it[dentro], minlength=len(X))
    cnt = np.bincount(k[dentro], minlength=len(X))
    y = np.where(cnt > 0, suma / np.maximum(cnt, 1), np.interp(X, tt, it))
    y = y - y.min()
    if y.max() <= 0:
        fuera["plano"] += 1; continue
    y = y / y.max()
    md = d.get("metadata"); md = json.loads(md) if isinstance(md, str) else (md or {})
    filas.append((y.astype(np.float16), SIST.index(cs), str(f.get("chemical_composition")), int(f.get("spacegroup") or 0),
                  n.split("/")[0]))
print("aceptados", len(filas)); print(dict(fuera))
perf, ys, comp, sg, origen = zip(*filas)
ys = np.array(ys)
print("por sistema", {SIST[k]: int(v) for k, v in enumerate(np.bincount(ys, minlength=7))})
print("por origen", collections.Counter(origen))
np.savez_compressed(RAW / "xrd_reales.npz", x=X.astype(np.float32), perfiles=np.stack(perf), y=ys,
                    composicion=np.array(comp), sg=np.array(sg), origen=np.array(origen), sistemas=np.array(SIST))
print("guardado")
