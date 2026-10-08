"""Tarea del Día 3: generar estructuras con MatterGen en la cola nukwa (GPU).

Se ejecuta desde tarea_mattergen.sbatch; no hace falta correrlo a mano.
Escribe en salidas_mattergen/<nombre>/:
  generated_crystals_cif.zip  (un CIF por estructura, lo lee el notebook 03)
  resumen.txt                 (fórmula, grupo espacial, distancia mínima, SMACT)
"""
import argparse, os, subprocess, sys, time, zipfile
from pathlib import Path

CURSO = Path(__file__).resolve().parents[1]
MODELOS = {"ninguna": "mattergen_base", "chemical_system": "chemical_system",
           "space_group": "space_group", "dft_band_gap": "dft_band_gap"}

p = argparse.ArgumentParser()
p.add_argument("--nombre", required=True, help="carpeta de salida en salidas_mattergen/")
p.add_argument("--condicion", default="ninguna", choices=list(MODELOS))
p.add_argument("--valor", default="", help="p. ej. Si-O-N, 225 o 1.5")
p.add_argument("--n", type=int, default=16, help="número de estructuras")
p.add_argument("--guia", type=float, default=2.0, help="factor de guía")
a = p.parse_args()

salida = CURSO / "salidas_mattergen" / a.nombre
salida.mkdir(parents=True, exist_ok=True)
_HF_PROPIO = Path(f"/work/{os.environ.get('USER', '')}/hf_cache")
hf = Path(os.environ.get("IA_CM_HF_HOME", _HF_PROPIO if _HF_PROPIO.is_dir() else "/work/trojas/hf_cache"))  # carpeta compartida del curso
cmd = [sys.executable, "-m", "mattergen.scripts.generate", str(salida),
       f"--pretrained-name={MODELOS[a.condicion]}", f"--batch_size={a.n}", "--num_batches=1",
       "--record_trajectories=False"]   # sin trayectorias: ahorra ~20 MB por lote
if a.condicion != "ninguna":
    valor = {"chemical_system": str, "space_group": int, "dft_band_gap": float}[a.condicion](a.valor)
    cmd += [f"--properties_to_condition_on={ {a.condicion: valor} }",
            f"--diffusion_guidance_factor={a.guia}"]
env = dict(os.environ, HF_HOME=str(hf), HF_HUB_OFFLINE="1",
           PYTHONPATH=f"{CURSO / 'mattergen_src'}:{os.environ.get('PYTHONPATH', '')}")

import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO HAY GPU", flush=True)
print("Pesos:", hf, "| comando:", " ".join(cmd[3:]), flush=True)
t0 = time.time()
r = subprocess.run(cmd, env=env)
print(f"MatterGen terminó con código {r.returncode} en {time.time() - t0:.0f} s", flush=True)
zip_cif = salida / "generated_crystals_cif.zip"
if r.returncode != 0 or not zip_cif.exists():
    sys.exit("⚠ No se generaron estructuras: revise el archivo .out de este trabajo.")

# resumen rápido para revisar sin abrir el notebook
from pymatgen.core import Structure
from smact.screening import smact_validity

filas = []
with zipfile.ZipFile(zip_cif) as z:
    for nombre in sorted(n for n in z.namelist() if n.endswith(".cif")):
        s = Structure.from_str(z.read(nombre).decode(), fmt="cif")
        f = s.composition.reduced_formula
        try:
            sg = s.get_space_group_info(symprec=0.1)
        except Exception:
            sg = ("?", 0)
        dmin = min(s.distance_matrix[i, j] for i in range(len(s)) for j in range(i + 1, len(s))) if len(s) > 1 else float("nan")
        try:
            ok = bool(smact_validity(f, include_alloys=False))
        except Exception:
            ok = False
        filas.append(f"{f:16s} {sg[0]:>10s} ({sg[1]:3d})  dmin={dmin:5.2f} Å  SMACT={'sí' if ok else 'no'}")
texto = (f"Condición: {a.condicion} = {a.valor or '-'}   guía = {a.guia}   n = {a.n}\n"
         f"Tiempo de MatterGen: {time.time() - t0:.0f} s\n\n" + "\n".join(filas) + "\n")
(salida / "resumen.txt").write_text(texto)
print(texto)
