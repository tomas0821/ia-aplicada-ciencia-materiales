"""Referencia para estimar E_hull con MatterSim en el notebook 03 (lo corre el instructor una vez, en GPU).

1. Referencia DFT: e_form mínima por fórmula reducida, de matbench_mp_e_form (Materials Project, 132 752 estructuras).
2. Potenciales químicos de MatterSim (mu por elemento), ajustados por mínimos cuadrados:
       E_MatterSim(estructura MP) - N * e_form_DFT = sum_i n_i * mu_i
   así e_form_MatterSim = (E_MatterSim - sum n_i mu_i) / N queda en la misma escala que la referencia DFT.
3. Validación con estructuras que no entraron al ajuste.

Salida en datos/hull/: referencia_mp_eform.json.gz, mu_mattersim.json
"""
import gzip, json, os, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import ase.constraints, ase.filters, ase.stress
for n in ["ExpCellFilter", "UnitCellFilter", "StrainFilter", "FrechetCellFilter"]:   # compatibilidad ASE 3.29
    setattr(ase.constraints, n, getattr(ase.filters, n))
ase.constraints.full_3x3_to_voigt_6_stress = ase.stress.full_3x3_to_voigt_6_stress
from pymatgen.core import Composition, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
import torch
from mattersim.forcefield import MatterSimCalculator

RAW = Path(os.environ.get("MB_EFORM", "/work/trojas/ia_cm_cv/raw/matbench_mp_e_form.json.gz"))
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("datos/hull")
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
filas = json.load(gzip.open(RAW))["data"]
print("estructuras", len(filas), f"{time.time() - t0:.0f} s", flush=True)


def comp_de(sd):
    c = Counter()
    for site in sd["sites"]:
        for sp in site["species"]:
            c[sp["element"]] += sp["occu"]
    return Composition(c)


comps = [comp_de(s) for s, _ in filas]
ef = np.array([e for _, e in filas])
ref = {}
for c, e in zip(comps, ef):
    f = c.reduced_formula
    if f not in ref or e < ref[f]:
        ref[f] = float(e)
with gzip.open(OUT / "referencia_mp_eform.json.gz", "wt") as g:
    json.dump(ref, g)
print("fórmulas de referencia", len(ref), flush=True)

# muestra para el ajuste: estructuras pequeñas, con todos los elementos bien representados
rng = np.random.default_rng(0)
cuenta, sel = Counter(), []
for i in rng.permutation(len(filas)):
    nat = len(filas[i][0]["sites"])
    if nat > 24:
        continue
    els = [str(e) for e in comps[i].elements]
    if any(cuenta[e] < 120 for e in els) or rng.random() < 0.03:
        sel.append(i); cuenta.update(els)
    if len(sel) >= 12000:
        break
print("muestra", len(sel), "elementos", len(cuenta), flush=True)

calc = MatterSimCalculator(device="cuda" if torch.cuda.is_available() else "cpu")
E = []
t1 = time.time()
for k, i in enumerate(sel):
    a = AseAtomsAdaptor.get_atoms(Structure.from_dict(filas[i][0])); a.calc = calc
    E.append(a.get_potential_energy())
    if k % 1000 == 0:
        print(k, f"{time.time() - t1:.0f} s", flush=True)
E = np.array(E)
# columnas: un potencial químico por elemento + una corrección por metal de transición "+U"
# cuando hay O o F (Materials Project mezcla cálculos GGA y GGA+U, como su esquema de correcciones)
U_EL = ["Co", "Cr", "Fe", "Mn", "Mo", "Ni", "V", "W"]
elementos = sorted(cuenta)
columnas = elementos + [f"U_{e}" for e in U_EL]
col = {e: j for j, e in enumerate(columnas)}


def fila_de(c):
    v = np.zeros(len(columnas)); d = c.get_el_amt_dict()
    for el, n in d.items():
        if el in col:
            v[col[el]] = n
    if "O" in d or "F" in d:
        for el in U_EL:
            if el in d:
                v[col[f"U_{el}"]] = d[el]
    return v


A = np.array([fila_de(comps[i]) for i in sel]); N = np.array([comps[i].num_atoms for i in sel])
b = E - N * ef[sel]
idx = rng.permutation(len(sel)); ntr = int(0.8 * len(sel)); tr, te = idx[:ntr], idx[ntr:]
mu, *_ = np.linalg.lstsq(A[tr], b[tr], rcond=None)
pred = (E[te] - A[te] @ mu) / N[te]
mae = float(np.mean(np.abs(pred - ef[sel][te])))
mu_full, *_ = np.linalg.lstsq(A, b, rcond=None)
print(f"MAE e_form (validación, {len(te)} estructuras): {mae:.3f} eV/átomo", flush=True)
conU = A[te][:, len(elementos):].sum(1) > 0
print(f"  con metal +U en óxido/fluoruro: {np.mean(np.abs(pred - ef[sel][te])[conU]):.3f} ({conU.sum()})  resto: {np.mean(np.abs(pred - ef[sel][te])[~conU]):.3f}", flush=True)
mediana = float(np.median(np.abs(pred - ef[sel][te])))
print(f"  mediana del error: {mediana:.3f}", flush=True)

# validación de E_hull: compuestos de validación que están sobre el casco de la referencia DFT
por_sis = defaultdict(list)
for f, e in ref.items():
    c = Composition(f)
    por_sis[tuple(sorted(str(x) for x in c.elements))].append((c, e))


def e_hull(c, e_form):
    els = sorted(str(x) for x in c.elements)
    entradas = [PDEntry(Composition(x), 0.0) for x in els]
    from itertools import combinations
    for r in range(1, len(els) + 1):
        for sub in combinations(els, r):
            entradas += [PDEntry(cc, ee * cc.num_atoms) for cc, ee in por_sis.get(tuple(sub), [])]
    pd = PhaseDiagram(entradas)
    return pd.get_e_above_hull(PDEntry(c, e_form * c.num_atoms), allow_negative=True)


eh = []
for r in te[:400]:
    i = sel[r]; c = comps[i]
    if len(c.elements) < 2:
        continue
    try:
        if abs(e_hull(c, ef[i])) < 1e-3:                       # estable en la referencia DFT
            eh.append(e_hull(c, float((E[r] - A[r] @ mu_full) / N[r])))
    except Exception:
        pass
eh = np.array(eh)
info = {"mu": dict(zip(columnas, map(float, mu_full))), "U_EL": U_EL, "mae_eform_eV_atomo": mae, "mediana_error_eform": mediana, "n_ajuste": len(sel),
        "validacion_ehull_estables": {"n": int(len(eh)), "mediana": float(np.median(eh)),
                                      "frac_menor_0.1": float(np.mean(eh < 0.1))},
        "modelo": "MatterSim-v1.0.0-1M", "fuente": "matbench_mp_e_form (Materials Project)"}
json.dump(info, open(OUT / "mu_mattersim.json", "w"), indent=1)
print(json.dumps({k: v for k, v in info.items() if k != "mu"}, indent=1), f"total {time.time() - t0:.0f} s")
