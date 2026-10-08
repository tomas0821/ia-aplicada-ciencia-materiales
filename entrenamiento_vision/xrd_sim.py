"""Simula patrones XRD (Cu Ka, 2θ 10–70°) para estructuras de matbench_mp_e_form.
Guarda posiciones e intensidades de picos (no el perfil) para que el entrenamiento
aplique el ensanchamiento y el ruido al vuelo (aumento de datos)."""
import os, sys, time, warnings
import numpy as np
from multiprocessing import Pool
from pathlib import Path

warnings.filterwarnings("ignore")
RAW = Path(os.environ.get("CV_RAW", "/work/trojas/ia_cm_cv/raw"))
N = int(os.environ.get("N_XRD", "60000"))
MAXP = 150
SISTEMAS = ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"]


def calc(args):
    i, d = args
    from pymatgen.core import Structure
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    try:
        s = Structure.from_dict(d)
        if s.num_sites > 60:
            return None
        sga = SpacegroupAnalyzer(s, symprec=0.1)
        cs = sga.get_crystal_system()
        sg = sga.get_space_group_number()
        p = XRDCalculator(wavelength="CuKa").get_pattern(s, scaled=True, two_theta_range=(10, 70))
        x, y = np.asarray(p.x, np.float32), np.asarray(p.y, np.float32)
        if len(x) == 0:
            return None
        o = np.argsort(-y)[:MAXP]
        pos = np.zeros(MAXP, np.float32); inte = np.zeros(MAXP, np.float32)
        pos[:len(o)] = x[o]; inte[:len(o)] = y[o]
        return i, s.composition.reduced_formula, sg, SISTEMAS.index(cs), pos, inte
    except Exception:
        return None


if __name__ == "__main__":
    from matminer.utils.io import load_dataframe_from_json
    t = time.time()
    df = load_dataframe_from_json(str(RAW / "matbench_mp_e_form.json.gz"))
    print("estructuras", len(df), f"{time.time()-t:.0f} s", flush=True)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(df), size=min(N, len(df)), replace=False)
    tareas = [(int(i), df["structure"].iloc[int(i)].as_dict()) for i in idx]
    res = []
    with Pool(int(os.environ.get("NPROC", "18"))) as pool:
        for k, r in enumerate(pool.imap_unordered(calc, tareas, chunksize=50)):
            if r is not None:
                res.append(r)
            if k % 5000 == 0:
                print(k, len(res), f"{time.time()-t:.0f} s", flush=True)
    ids, formulas, sgs, ys, pos, inte = zip(*res)
    np.savez_compressed(RAW / "xrd_picos.npz", idx=np.array(ids), formula=np.array(formulas), sg=np.array(sgs),
                        y=np.array(ys), pos=np.stack(pos), inte=np.stack(inte), sistemas=np.array(SISTEMAS))
    print("guardado", len(res), np.bincount(np.array(ys), minlength=7), f"{time.time()-t:.0f} s")
