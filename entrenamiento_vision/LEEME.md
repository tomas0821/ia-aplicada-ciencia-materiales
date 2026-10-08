# Entrenamiento de los modelos de visión computacional (Día 3, P3.5)

Los estudiantes NO corren estos scripts: reciben los pesos en `datos/vision/modelos/`.
Se dejan aquí para reproducir el entrenamiento (Kabré, cola nukwa, 1 GPU L40S, 12 núcleos).

Orden:
1. `descargas.py`: matbench_mp_e_form (estructuras de MP), opXRD (Zenodo) y NFFA-Europe SEM (B2SHARE, 450 imágenes al azar por categoría).
2. `xrd_sim.py`: simula los XRD (Cu Kα, 2θ 10–70°) de 60 000 estructuras con pymatgen (unos 7 min con 18 procesos).
3. `prep_opxrd.py`: extrae los patrones experimentales de una sola fase, Cu Kα y rango 10–70°.
4. `train_xrd.py`: CNN 1D con aumento de datos al vuelo (unos 9 min en GPU).
5. `train_sem.py`: ajuste fino de ResNet-18 (ImageNet) con las imágenes SEM.
6. `empaquetar.py`: arma `datos/vision/` para el curso.

Rutas por defecto en `/work/$USER/ia_cm_cv`; se cambian con las variables `CV_RAW` y `CV_OUT`.

Nota: en la corrida del 23/09/2026 las imágenes se recortaron en dos pasos (10 % inferior al
descargar y luego `fix_sem.py`, que quita otros ~15 %), porque la franja de metadatos del SEM
seguía visible y el modelo podría usarla como atajo. `descargas.py` ya recorta el 20 % inferior.
