# Tareas con GPU (cola nukwa de Kabré)

En clase se usan modelos y estructuras ya preparados. Estas dos tareas se envían
a la cola de GPU y se revisan después: no hay que esperar frente a la pantalla.

| Tarea | Archivo | Qué hace | Tiempo de GPU |
|---|---|---|---|
| Día 3, MatterGen | `tarea_mattergen.sbatch` | Genera 16 estructuras con la condición que usted elija | unos 5 min |
| Día 3, visión | `tarea_vision.sbatch` | Entrena la CNN de XRD 20 épocas con una variante del aumento de datos | unos 5 min |

## Pasos

1. Abra el `.sbatch` y cambie solo las líneas entre `==== Edite solo estas líneas ====`.
2. Envíelo desde el notebook (celda "Enviar la tarea") o desde una terminal, en la carpeta del curso:
   `sbatch tareas/tarea_mattergen.sbatch`
3. Revise el estado con `estado_gpu()` en el notebook o `squeue -u $USER` (PENDING = esperando GPU, RUNNING = corriendo).
   Solo se permite **un trabajo de GPU a la vez** por estudiante: espere a que termine antes de enviar otro.
4. El registro queda en `tareas/registros/` y los resultados en `salidas_mattergen/<NOMBRE>/`
   o `salidas_vision/<NOMBRE>/`. Los notebooks 03 y 06 tienen una celda para cargarlos.

## Notas para el instructor

- Cola `nukwa-l40s` (tope de 4 h). Cada trabajo pide 10 de los 20 núcleos del nodo, así caben 2 por GPU
  y la cola reparte los trabajos entre los 7 nodos L40S en vez de amontonarlos en uno. La cola `nukwa`
  asigna el nodo completo a cada trabajo.
- Prueba del 26 set 2026 (nukwa-05, 4 generaciones de 16 estructuras en la misma GPU): compartir la GPU
  no acelera, solo reparte el tiempo. Cada generación cuesta unos 4 min de GPU (sin condición 6.5 min con
  4 compartiendo; con condición 12 a 15 min). Para 50 estudiantes: unos 200 min de GPU, del orden de
  30 a 60 min de cola si hay 6 o 7 nodos L40S libres.
- Entrenamiento de visión (20 épocas): 1 a 2 min por trabajo. Aumento completo: 51 % en simulados y
  29 % en reales; sin aumento: 15 % y 9 %.
- Un trabajo a la vez por estudiante: `tareas/gpu.py` no envía si ya hay uno en la cola, y los dos .sbatch usan el
  nombre `ia_cm_gpu` con `--dependency=singleton`, así SLURM corre solo uno por persona aunque envíen varios desde
  una terminal. El límite estricto (que rechace el envío) solo lo puede poner el CNCA con una QOS para las cuentas
  del curso (MaxSubmitJobsPerUser=1).
- Límite de Kabré: 4 trabajos en ejecución por usuario en nukwa; los demás esperan en cola.
- Pesos de MatterGen: variable `IA_CM_HF_HOME` (por defecto `/work/$USER/hf_cache`). Con una carpeta
  compartida de solo lectura del CNCA basta con cambiar la línea `export IA_CM_HF_HOME=...`.
- `generar_mattergen.py` usa el código de `mattergen_src/`, igual que el notebook 03.
- `entrenar_xrd.py` usa `datos/vision/xrd_picos.npz` (listas de picos simuladas, 34 MB) y la misma
  partición por fórmula que el modelo del curso.
