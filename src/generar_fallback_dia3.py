"""Genera estructuras placeholder para el modo fallback del Día 3.

⚠ PLACEHOLDER: estas son estructuras prototipo conocidas con perturbaciones,
NO salidas de MatterGen. Sirven para probar el pipeline de análisis/filtrado
del notebook 03 sin GPU. Cuando se corra MatterGen en el servidor GPU
(tarea pendiente), sus CIFs reales deben reemplazar estos archivos en las
mismas carpetas: datos/mattergen_fallback/{incondicional,limnox,solar,cubico}/
"""

from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure

rng = np.random.default_rng(7)

BASE = Path(__file__).resolve().parent.parent / "datos" / "mattergen_fallback"


def jitter(a: float) -> float:
    return float(a * (1 + rng.uniform(-0.04, 0.04)))


def rocksalt(a, A, B):
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), [A, B], [[0, 0, 0], [0.5, 0.5, 0.5]])


def perovskite(a, A, B, X):
    return Structure.from_spacegroup(
        "Pm-3m", Lattice.cubic(a), [A, B, X], [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]
    )


def fluorite(a, A, X):
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), [A, X], [[0, 0, 0], [0.25, 0.25, 0.25]])


def wurtzite(a, c, A, X):
    lat = Lattice.hexagonal(a, c)
    return Structure.from_spacegroup(
        "P6_3mc", lat, [A, X], [[1 / 3, 2 / 3, 0], [1 / 3, 2 / 3, 0.375]]
    )


def rutile(a, c, A, X):
    lat = Lattice.tetragonal(a, c)
    return Structure.from_spacegroup(
        "P4_2/mnm", lat, [A, X], [[0, 0, 0], [0.305, 0.305, 0]]
    )


LOTES = {
    "incondicional": [
        lambda: rocksalt(jitter(4.33), "Ti", "C"),
        lambda: rocksalt(jitter(4.24), "Ti", "N"),
        lambda: rutile(jitter(4.59), jitter(2.96), "Ti", "O"),
        lambda: perovskite(jitter(3.91), "Sr", "Ti", "O"),
        lambda: fluorite(jitter(5.46), "Ca", "F"),
        lambda: wurtzite(jitter(3.25), jitter(5.21), "Zn", "O"),
        lambda: rocksalt(jitter(4.21), "Mg", "O"),
        lambda: rocksalt(jitter(5.64), "Na", "Cl"),
        lambda: perovskite(jitter(4.00), "Ba", "Ti", "O"),
        lambda: rocksalt(jitter(4.46), "Ta", "C"),
    ],
    "limnox": [
        lambda: rocksalt(jitter(4.44), "Mn", "O"),
        lambda: fluorite(jitter(4.61), "Li", "O"),  # antifluorita Li2O aprox
        lambda: perovskite(jitter(3.88), "Li", "Mn", "O"),
        lambda: rocksalt(jitter(4.10), "Li", "Mn"),
        lambda: rutile(jitter(4.40), jitter(2.87), "Mn", "O"),
    ],
    "solar": [
        lambda: wurtzite(jitter(3.82), jitter(6.26), "Cd", "S"),
        lambda: rocksalt(jitter(5.94), "Pb", "S"),
        lambda: wurtzite(jitter(3.19), jitter(5.19), "Ga", "N"),
        lambda: fluorite(jitter(6.48), "Cd", "Te"),  # aprox zinc-blenda por fluorita
        lambda: perovskite(jitter(6.00), "Cs", "Pb", "I"),
    ],
    "cubico": [
        lambda: rocksalt(jitter(4.33), "Ti", "C"),
        lambda: perovskite(jitter(3.91), "Sr", "Ti", "O"),
        lambda: fluorite(jitter(5.46), "Ca", "F"),
        lambda: rocksalt(jitter(4.21), "Mg", "O"),
        lambda: perovskite(jitter(4.00), "Ba", "Ti", "O"),
    ],
}

N_POR_LOTE = {"incondicional": 10, "limnox": 20, "solar": 20, "cubico": 20}

for lote, protos in LOTES.items():
    carpeta = BASE / lote
    carpeta.mkdir(parents=True, exist_ok=True)
    n = N_POR_LOTE[lote]
    for i in range(n):
        s = protos[i % len(protos)]()
        s.to(filename=str(carpeta / f"{i:02d}_{s.composition.reduced_formula}.cif"))
    print(f"{lote}: {n} CIFs en {carpeta}")

(BASE / "README.txt").write_text(
    "PLACEHOLDER — estructuras prototipo perturbadas para probar el pipeline\n"
    "del Día 3 sin GPU. Reemplazar con CIFs reales de MatterGen generados en\n"
    "el servidor GPU (misma estructura de carpetas).\n"
)
print("README escrito.")
