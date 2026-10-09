# IA Aplicada a Ciencia de Materiales

Notebooks, datos y tareas del curso intensivo **IA Aplicada a Ciencia de Materiales**
(CeNAT y CNCA, Costa Rica, 5 al 9 de octubre de 2026). El curso corre en
**Kabré OnDemand** con el kernel **Python-AI-Materials**, y también hay versiones para **Google Colab**.

## Contenido

| Día | Notebook | Tema |
|---|---|---|
| 1 | `notebooks/00_test_entorno.ipynb` | Revisar que el entorno funciona |
| 1 | `notebooks/01_exploracion_datos.ipynb` | Explorar datos de materiales, estructuras y descriptores Magpie |
| 2 | `notebooks/02_sklearn_matminer.ipynb` | Predecir propiedades con Random Forest, screening y SHAP |
| 3 | `notebooks/03_mattergen_generacion.ipynb` | Generar cristales con MatterGen, filtrarlos y medir su estabilidad con MatterSim |
| 3 | `notebooks/06_vision_computacional.ipynb` | Visión computacional: XRD y SEM con modelos ya entrenados |
| 4 | `notebooks/04_doe_bayesiana.ipynb` | Diseño de experimentos y optimización bayesiana |
| 5 | `notebooks/07_mas_alla_del_curso.ipynb` | Otros datasets de materiales y cómo empezar con los suyos |
| 5 | `notebooks/05_brief_A_termoelectrico.ipynb`, `05_brief_B_recubrimiento.ipynb` | Mini reto final |

Además:

- `notebooks/soluciones/`: soluciones de los retos y ejercicios.
- `src/`: versiones en script de los notebooks y utilidades.
- `tareas/`: trabajos cortos para la GPU de Kabré (generación con MatterGen y entrenamiento de visión). Ver `tareas/LEEME.md`.
- `datos/`: datasets pequeños, cachés y modelos ya entrenados que usan los notebooks.
- `presentaciones/`: las presentaciones de los cinco días y la de conceptos, en PDF.
- `mattergen_src.zip`: código fuente de MatterGen v1.0.3 (Microsoft, licencia MIT; ver `mattergen_src/LICENSE` dentro del zip). Descomprímalo en la raíz (`unzip mattergen_src.zip`) para tener la carpeta `mattergen_src/`. Incluye un cambio de una línea en `eval_utils.py` para que los CIF se escriban en la carpeta de salida y no en `/tmp`.
- `entrenamiento_vision/`: scripts con que se entrenaron los modelos de visión.
- `NOTAS_TECNICAS.md`: notas de instalación y pruebas en Kabré.

## Cómo usarlo en Kabré

1. Abra una sesión de **Jupyter** en Kabré OnDemand, cola **kura**.
2. Copie esta carpeta a su home (por ejemplo `~/curso`), descargándola de GitHub o copiándola de la carpeta del curso en Kabré.
3. Abra el notebook del día y elija el kernel **Python-AI-Materials** (Kernel → Change kernel).
4. Corra las celdas de arriba hacia abajo.

## Cómo usarlo en Google Colab

Las versiones de `notebooks/colab/` corren en [Google Colab](https://colab.research.google.com) sin instalar nada en su computadora.

| Notebook | Tema | Abrir |
|---|---|---|
| `00_test_entorno` | Revisar el entorno | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tomas0821/ia-aplicada-ciencia-materiales/blob/main/notebooks/colab/00_test_entorno.ipynb) |
| `01_exploracion_datos` | Explorar datos y descriptores | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tomas0821/ia-aplicada-ciencia-materiales/blob/main/notebooks/colab/01_exploracion_datos.ipynb) |
| `02_sklearn_matminer` | Predecir propiedades | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tomas0821/ia-aplicada-ciencia-materiales/blob/main/notebooks/colab/02_sklearn_matminer.ipynb) |
| `03_mattergen_generacion` | Generar cristales con MatterGen | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tomas0821/ia-aplicada-ciencia-materiales/blob/main/notebooks/colab/03_mattergen_generacion.ipynb) |
| `04_doe_bayesiana` | DoE y optimización bayesiana | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tomas0821/ia-aplicada-ciencia-materiales/blob/main/notebooks/colab/04_doe_bayesiana.ipynb) |
| `05_brief_A_termoelectrico` | Mini reto: Brief A | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tomas0821/ia-aplicada-ciencia-materiales/blob/main/notebooks/colab/05_brief_A_termoelectrico.ipynb) |
| `05_brief_B_recubrimiento` | Mini reto: Brief B | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tomas0821/ia-aplicada-ciencia-materiales/blob/main/notebooks/colab/05_brief_B_recubrimiento.ipynb) |
| `06_vision_computacional` | Visión computacional | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tomas0821/ia-aplicada-ciencia-materiales/blob/main/notebooks/colab/06_vision_computacional.ipynb) |
| `07_mas_alla_del_curso` | Más allá del curso | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tomas0821/ia-aplicada-ciencia-materiales/blob/main/notebooks/colab/07_mas_alla_del_curso.ipynb) |

1. Abra el notebook con el botón **Abrir en Colab** (necesita una cuenta de Google).
2. Corra primero la celda ☁️: baja este repositorio, reconstruye los archivos grandes e instala las librerías que Colab no trae. Tarda de 1 a 4 minutos.
3. Siga de arriba hacia abajo, como en Kabré.

Notas:

- Cada notebook de Colab es una máquina nueva. El 02 rehace por su cuenta la tabla que guarda el 01 (uno o dos minutos más).
- **GPU:** para el 03 (MatterGen) y el 06 (entrenamiento de visión) conviene *Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU T4*. Sin GPU, el 03 usa estructuras ya generadas y el 06 se salta solo el entrenamiento propio (el resto funciona).
- **MatterGen de verdad en Colab:** en el 03, ponga `USAR_MATTERGEN = True` en la celda ⚙️. Los pesos (unos 1.9 GB, de Hugging Face) se guardan en su Google Drive, en la carpeta `ia_cm_hf_cache`, así que se descargan solo la primera vez.
- La relajación con MatterSim del 03 tarda varios minutos en los 2 núcleos de Colab.
- Lo que guarde en Colab se borra al cerrar la sesión: descargue sus resultados o guárdelos en Drive.

## Antes de usarlo fuera de Kabré (sin Colab)

Para que GitHub aceptara la subida por la web, tres carpetas van comprimidas y tres archivos grandes van partidos en trozos (`.parte0`, `.parte1`, ...). Arréglelos una vez desde la raíz del repo (la celda ☁️ de Colab lo hace sola):

```bash
unzip mattergen_src.zip
unzip datos/mattergen_fallback.zip -d datos
unzip datos/vision/sem_prueba.zip -d datos/vision
cat datos/hull/mattersim-v1.0.0-1M.pth.parte{0,1} > datos/hull/mattersim-v1.0.0-1M.pth
cat datos/vision/xrd_picos.npz.parte{0,1,2,3} > datos/vision/xrd_picos.npz
cat datos/vision/modelos/sem_resnet18.pt.parte{0,1,2} > datos/vision/modelos/sem_resnet18.pt
```

## Lo que no viene en el repo

- **Pesos de MatterGen** (unos 1.9 GB): están en Kabré, en la carpeta compartida del curso.
  El notebook 03 los busca en `/work/$USER/hf_cache` y, si no existen, en la carpeta compartida.
  En Colab se bajan de Hugging Face y quedan en su Google Drive (ver arriba).
  Sin GPU o sin pesos, el notebook usa estructuras ya generadas (`datos/mattergen_fallback/`)
  y todo el análisis funciona igual.
- **Clave de Materials Project**: no hace falta en clase, porque los datos vienen guardados en `datos/`.
  Para usar la API en casa, cree un archivo `.env` con `MP_API_KEY=su_clave` (no lo suba al repo).

## Datos de terceros

Los conjuntos de datos incluidos conservan la licencia de su fuente original.
Los detalles de los datos de visión (Materials Project, opXRD y NFFA-Europe, CC BY 4.0)
están en `datos/vision/LEEME.md`. Los datasets de `datos/matminer_cache/` vienen de
[matminer](https://hackingmaterials.lbl.gov/matminer/) y Materials Project.
