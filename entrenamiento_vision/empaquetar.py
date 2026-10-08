"""Arma la carpeta datos/vision del curso a partir de los resultados del entrenamiento."""
import json, os, shutil
from pathlib import Path
import numpy as np

B = Path("/work/trojas/ia_cm_cv")
RAW, OUT = B / "raw", B / "out"
DEST = B / "vision"
shutil.rmtree(DEST, ignore_errors=True)
(DEST / "modelos").mkdir(parents=True)
for f in ["xrd_cnn1d.pt", "xrd_info.json", "sem_resnet18.pt", "sem_info.json"]:
    shutil.copy(OUT / f, DEST / "modelos" / f)
shutil.copy(OUT / "xrd_simulados_prueba.npz", DEST / "xrd_simulados_prueba.npz")
shutil.copy(RAW / "xrd_reales.npz", DEST / "xrd_reales_opxrd.npz")
split = json.load(open(OUT / "sem_split.json"))
for f, k in split["test"]:
    c = split["clases"][k]
    (DEST / "sem_prueba" / c).mkdir(parents=True, exist_ok=True)
    shutil.copy(f, DEST / "sem_prueba" / c / Path(f).name)

info_x = json.load(open(OUT / "xrd_info.json")); info_s = json.load(open(OUT / "sem_info.json"))
(DEST / "LEEME.md").write_text(f"""# Datos y modelos de la práctica de visión computacional (Día 3, P3.5)

## Contenido
- `xrd_simulados_prueba.npz`: {len(np.load(DEST / 'xrd_simulados_prueba.npz')['y'])} difractogramas simulados (Cu Kα, 2θ 10–70°, 1200 puntos) que el modelo no vio al entrenar.
- `xrd_reales_opxrd.npz`: {len(np.load(DEST / 'xrd_reales_opxrd.npz')['y'])} difractogramas experimentales de una sola fase, tomados de opXRD e interpolados a la misma rejilla.
- `sem_prueba/`: 20 imágenes SEM por categoría (NFFA-Europe), reducidas a 256 × 256 píxeles en gris y sin la franja de metadatos.
- `modelos/xrd_cnn1d.pt`: red convolucional 1D (sistema cristalino). Exactitud en simulados de prueba: {info_x['acc_test_ruido']:.1%}; en reales: {info_x.get('acc_reales', float('nan')):.1%}.
- `modelos/sem_resnet18.pt`: ResNet-18 ajustada a 10 categorías SEM. Exactitud en prueba: {info_s['test_acc']:.1%}.

## Origen y licencias (todas CC BY 4.0)
- Estructuras: matbench_mp_e_form (Dunn et al. 2020, npj Comput. Mater. 6, 138), derivado de Materials Project (Jain et al. 2013).
- XRD experimentales: opXRD, Hollarek et al. (2025), Zenodo 10.5281/zenodo.14314251.
- SEM: Aversa et al. (2018), Sci. Data 5, 180172; datos en B2SHARE 10.23728/b2share.80df8606fcdb4b2bae1656f0dc6db8ba.
- Pesos iniciales de ResNet-18: torchvision (ImageNet).

Los modelos se entrenaron en Kabré (nukwa, GPU L40S) con los scripts de `curso/entrenamiento_vision/`.
""")
tot = sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file())
print("listo", DEST, f"{tot/1e6:.1f} MB")
