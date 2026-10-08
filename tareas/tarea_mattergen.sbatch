#!/bin/bash
# Tarea del Día 3: generar estructuras con MatterGen en un nodo con GPU.
# Envío (desde la carpeta del curso):   sbatch tareas/tarea_mattergen.sbatch
# Estado:                               squeue -u $USER
# Un solo trabajo de GPU a la vez por estudiante (ver tareas/gpu.py).
# Resultado: salidas_mattergen/<NOMBRE>/ (zip de CIF y resumen.txt)
#
#SBATCH --job-name=ia_cm_gpu            # mismo nombre en las dos tareas
#SBATCH --partition=nukwa-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=tareas/registros/mattergen_%j.out
#SBATCH --dependency=singleton          # un solo trabajo ia_cm_gpu a la vez por persona

# ======== Edite solo estas líneas ========
NOMBRE="mi_tarea"                 # carpeta de salida (sin espacios)
CONDICION="chemical_system"       # chemical_system | space_group | dft_band_gap | ninguna
VALOR="Si-O-N"                    # p. ej. "Li-Fe-P-O", "225" o "1.5"
N=16                              # número de estructuras (máximo 20)
GUIA=2.0                          # factor de guía
# =========================================

# pesos de MatterGen (carpeta compartida del curso o copia propia en /work)
if [ -z "$IA_CM_HF_HOME" ]; then
  if [ -d "/work/$USER/hf_cache" ]; then IA_CM_HF_HOME="/work/$USER/hf_cache"; else IA_CM_HF_HOME="/work/trojas/hf_cache"; fi
fi
export IA_CM_HF_HOME
PY=/opt/python/mamba3/envs/python-AI-materials/bin/python

cd "$SLURM_SUBMIT_DIR"
echo "Nodo: $(hostname)  Inicio: $(date)"
$PY tareas/generar_mattergen.py --nombre "$NOMBRE" --condicion "$CONDICION" \
    --valor "$VALOR" --n "$N" --guia "$GUIA"
echo "Fin: $(date)"
