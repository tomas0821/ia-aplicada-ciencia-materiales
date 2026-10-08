#!/bin/bash
# Tarea de visión computacional: entrenamiento corto de la CNN de XRD en GPU.
# Envío (desde la carpeta del curso):   sbatch tareas/tarea_vision.sbatch
# Estado:                               squeue -u $USER
# Un solo trabajo de GPU a la vez por estudiante (ver tareas/gpu.py).
# Resultado: salidas_vision/<NOMBRE>/ (pesos e info.json)
#
#SBATCH --job-name=ia_cm_gpu            # mismo nombre en las dos tareas
#SBATCH --partition=nukwa-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=tareas/registros/vision_%j.out
#SBATCH --dependency=singleton          # un solo trabajo ia_cm_gpu a la vez por persona

# ======== Edite solo estas líneas ========
NOMBRE="sin_fondo"          # carpeta de salida (sin espacios)
AUMENTO="sin_fondo"         # completo | ninguno | sin_fondo | sin_kalfa2 | sin_ruido | sin_ensanchamiento
EPOCAS=20                   # 20 épocas ≈ unos minutos en GPU
FRACCION=1.0                # fracción de los ~45 000 patrones de entrenamiento
# =========================================

PY=/opt/python/mamba3/envs/python-AI-materials/bin/python
cd "$SLURM_SUBMIT_DIR"
echo "Nodo: $(hostname)  Inicio: $(date)"
$PY tareas/entrenar_xrd.py --nombre "$NOMBRE" --aumento "$AUMENTO" \
    --epocas "$EPOCAS" --fraccion "$FRACCION"
echo "Fin: $(date)"
