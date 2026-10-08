# Datos y modelos de la práctica de visión computacional (Día 3, P3.5)

## Contenido
- `xrd_simulados_prueba.npz`: 700 difractogramas simulados (Cu Kα, 2θ 10–70°, 1200 puntos) que el modelo no vio al entrenar.
- `xrd_reales_opxrd.npz`: 328 difractogramas experimentales de una sola fase, tomados de opXRD e interpolados a la misma rejilla.
- `sem_prueba/`: 20 imágenes SEM por categoría (NFFA-Europe), reducidas a 256 × 256 píxeles en gris y sin la franja de metadatos.
- `modelos/xrd_cnn1d.pt`: red convolucional 1D (sistema cristalino). Exactitud en simulados de prueba: 59.1%; en reales: 27.7%.
- `modelos/sem_resnet18.pt`: ResNet-18 ajustada a 10 categorías SEM. Exactitud en prueba: 93.5%.

## Origen y licencias (todas CC BY 4.0)
- Estructuras: matbench_mp_e_form (Dunn et al. 2020, npj Comput. Mater. 6, 138), derivado de Materials Project (Jain et al. 2013).
- XRD experimentales: opXRD, Hollarek et al. (2025), Zenodo 10.5281/zenodo.14314251.
- SEM: Aversa et al. (2018), Sci. Data 5, 180172; datos en B2SHARE 10.23728/b2share.80df8606fcdb4b2bae1656f0dc6db8ba.
- Pesos iniciales de ResNet-18: torchvision (ImageNet).

Los modelos se entrenaron en Kabré (nukwa, GPU L40S) con los scripts de `curso/entrenamiento_vision/`.
