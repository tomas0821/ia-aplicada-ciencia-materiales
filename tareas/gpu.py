"""Envío de trabajos a la GPU de Kabré (cola nukwa-l40s): un trabajo a la vez por estudiante.

Dos protecciones:
1. enviar_a_gpu() no envía si usted ya tiene un trabajo del curso en la cola.
2. Los .sbatch usan el mismo nombre de trabajo (ia_cm_gpu) con --dependency=singleton:
   aunque alguien envíe varios desde una terminal, SLURM corre solo uno a la vez por persona.
"""
import os
import subprocess
from pathlib import Path

CURSO = Path(__file__).resolve().parents[1]
NOMBRE_TRABAJO = "ia_cm_gpu"          # el mismo en tarea_mattergen.sbatch y tarea_vision.sbatch


def mis_trabajos():
    """Trabajos del curso que usted tiene en la cola (esperando o corriendo)."""
    r = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "--name", NOMBRE_TRABAJO,
                        "-o", "%i %T %M %R"], capture_output=True, text=True)
    return [linea for linea in r.stdout.splitlines() if linea.strip()]


def enviar_a_gpu(script):
    """Envía el .sbatch solo si no hay otro trabajo suyo en la GPU. Devuelve el número de trabajo."""
    activos = mis_trabajos()
    if activos:
        print("⏳ Ya tiene un trabajo en la GPU. Espere a que termine antes de enviar otro:")
        for linea in activos:
            print("   ", linea)
        return None
    (CURSO / "tareas" / "registros").mkdir(exist_ok=True)
    r = subprocess.run(["sbatch", "--parsable", script], cwd=CURSO, capture_output=True, text=True)
    if r.returncode != 0:
        print("⚠ No se pudo enviar:", r.stderr.strip())
        return None
    jid = r.stdout.strip().split(";")[0]
    print(f"✓ Trabajo {jid} enviado. Revise en qué va con estado_gpu().")
    return jid


def estado_gpu():
    """PENDING = esperando GPU, RUNNING = corriendo. Si no aparece nada, ya terminó."""
    activos = mis_trabajos()
    if activos:
        print("TRABAJO  ESTADO  TIEMPO  DETALLE")
        print("\n".join(activos))
    else:
        print("No tiene trabajos en la GPU (si envió uno, ya terminó: revise tareas/registros/).")
