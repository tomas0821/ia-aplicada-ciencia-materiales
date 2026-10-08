# %% [markdown]
# <div style="background:linear-gradient(90deg,#1A2E4A 0%,#2E5E9B 100%);color:white;padding:20px 26px;border-radius:12px;">
# <span style="font-size:12px;letter-spacing:3px;color:#D6E8F7;">CURSO IA APLICADA A CIENCIA DE MATERIALES &nbsp;·&nbsp; DÍA 4 · OPTIMIZACIÓN</span><br>
# <span style="font-size:26px;font-weight:700;">⚗️ Diseño de Experimentos y Optimización Bayesiana</span><br>
# <span style="font-size:14px;color:#D6E8F7;">¿Qué experimento conviene hacer a continuación?</span>
# </div>
#
# | Bloque | Contenido | Tiempo |
# |--------|-----------|--------|
# | P4.1 | Diseño de Experimentos con pyDOE3 (demo dentro de T4.3) | 25 min |
# | P4.2 | Optimización Bayesiana (Hartmann 6D) | 45 min |
# | P4.3 | BO sobre un proceso PVD (simulado) | 25 min |
# | P4.4 | Discusión en grupo | 5 min |
#
# El Reto 3 y el explorador interactivo de P4.3 quedan como opcionales o para casa.
#
# **La idea central de hoy:** cada experimento cuesta tiempo y dinero. La
# pregunta no es "¿qué pasa si pruebo X?", sino **"¿qué experimento me
# conviene hacer a continuación?"**. Primero lo respondemos con diseños
# estadísticos clásicos (DoE) y luego con un agente que aprende de cada
# resultado: la **Optimización Bayesiana** — el cerebro de los laboratorios
# autónomos que vimos en teoría (Kusne & Takeuchi 2020, MacLeod 2020).

# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>P4.1 — Diseño de Experimentos con pyDOE3</b> <span style="float:right;background:#E87C1E;color:white;padding:2px 12px;border-radius:12px;font-size:12px;font-weight:600;">⏱ 25 min</span></div>
#
# Escenario: depositamos recubrimientos duros (Ti,Al)N por PVD y controlamos
# 4 variables de proceso. ¿Dónde poner los primeros 20 experimentos?


# %%
# --- 🎨 Estilo visual del curso (paleta validada para accesibilidad) ---
import matplotlib.pyplot as plt
from cycler import cycler

AZUL, NARANJA, PURPURA, VERDE = "#2E5E9B", "#D96F0F", "#B0509E", "#2F9E44"
TINTA, GRIS = "#1A2E4A", "#8A94A3"
plt.rcParams.update({
    "axes.prop_cycle": cycler(color=[AZUL, NARANJA, PURPURA, VERDE]),  # orden fijo
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": TINTA, "axes.labelcolor": TINTA, "text.color": TINTA,
    "xtick.color": TINTA, "ytick.color": TINTA,
    "axes.titleweight": "bold", "axes.titlesize": 13, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
    "lines.linewidth": 2.0, "figure.dpi": 105, "legend.frameon": False,
})

# --- Opciones de visualización (Kabré OnDemand) ---
# plotly (gráficos interactivos) y nglview (visor 3D) necesitan que el servidor
# Jupyter vea sus extensiones. activar_extensiones() las enlaza en su carpeta
# personal (~/.local) y ajusta la versión del visor 3D. Si imprime un aviso,
# recargue la página (F5) una sola vez y siga. Si algo falla, ponga ambas
# banderas en False: se usan matplotlib y ASE.
USAR_PLOTLY = True
USAR_NGLVIEW = True


def activar_extensiones():
    import json, sys
    from pathlib import Path
    fuente = Path(sys.prefix) / "share" / "jupyter" / "labextensions"
    destino = Path.home() / ".local" / "share" / "jupyter" / "labextensions"
    nuevas = []
    for ext in ["jupyterlab-plotly", "nglview-js-widgets"]:
        if (fuente / ext).is_dir() and not (destino / ext).exists():
            destino.mkdir(parents=True, exist_ok=True)
            (destino / ext).symlink_to(fuente / ext)
            nuevas.append(ext)
    if nuevas:
        print("Extensiones activadas:", ", ".join(nuevas))
        print("👉 Recargue la página (F5) una vez y continúe.")
    paquete = fuente / "nglview-js-widgets" / "package.json"
    if not (USAR_NGLVIEW and paquete.is_file()):
        return
    # nglview 4.0.1 pide la parte JavaScript "4.0", pero el paquete instalado
    # trae la 3.1.5 (error de empaquetado). Pedimos la versión que sí existe.
    import ipywidgets
    with ipywidgets.Output():
        import nglview  # noqa: F401  (al importarse muestra un widget; lo ocultamos)
    version_js = json.loads(paquete.read_text())["version"]

    def subclases(c):
        for s in c.__subclasses__():
            yield s
            yield from subclases(s)

    for c in set(subclases(ipywidgets.Widget)):
        t = c.class_traits()
        if "_model_module" in t and t["_model_module"].default_value == "nglview-js-widgets":
            for n in ("_model_module_version", "_view_module_version"):
                if n in t:
                    t[n].default_value = version_js
                    iniciales = getattr(c, "_static_immutable_initial_values", None)
                    if iniciales is not None and n in iniciales:
                        iniciales[n] = version_js


activar_extensiones()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pyDOE3 import lhs, bbdesign, pbdesign

def encontrar_datos() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents[:3]]:
        if (base / "datos").is_dir():
            return base / "datos"
    raise FileNotFoundError("No encuentro la carpeta datos/")

DATOS = encontrar_datos()

# %% [markdown]
# ### Latin Hypercube Sampling (LHS)
#
# Reparte N puntos de modo que **cada variable quede uniformemente muestreada**
# — sin las esquinas redundantes del factorial completo. Es el diseño estándar
# para *inicializar* una campaña cuando aún no sabes nada del proceso.

# %%
n_vars = 4
n_samples = 20
X_lhs = lhs(n_vars, samples=n_samples, criterion="maximin", seed=42)

# Escalar de [0,1] a los rangos reales del proceso PVD
var_names = ["T_sustrato (°C)", "P_N2 (Pa)", "bias (V)", "flujo Ti/Al"]
var_min = np.array([300, 0.3, -200, 0.5])
var_max = np.array([550, 1.5, -50, 2.0])
X_real = X_lhs * (var_max - var_min) + var_min

df_lhs = pd.DataFrame(X_real, columns=var_names)
print(f"Diseño LHS: {df_lhs.shape[0]} experimentos × {df_lhs.shape[1]} variables")
df_lhs.head()

# %% [markdown]
# <div style="background:#EEF3F9;border-left:6px solid #2E5E9B;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#2E5E9B;">🔍 Cómo funciona este código</b></div>
#
# - `lhs(4, samples=20)` devuelve una matriz 20×4 con valores en [0, 1]: divide
#   cada variable en 20 franjas y pone exactamente un punto en cada franja.
# - `criterion="maximin"` prueba varios diseños y se queda con el que maximiza
#   la distancia mínima entre puntos (evita puntos amontonados).
# - `seed=42` fija el azar. Las versiones nuevas de pyDOE3 usan `seed`; el
#   antiguo `random_state` genera un aviso.
# - `X_lhs * (var_max - var_min) + var_min` es un cambio de escala columna a
#   columna gracias al *broadcasting* de NumPy: pasa de [0, 1] al rango real.

# %%
pd.plotting.scatter_matrix(df_lhs, alpha=0.8, figsize=(9, 9), diagonal="hist")
plt.suptitle("LHS: cobertura uniforme del espacio experimental", y=1.01)
plt.show()

# 🤔 Miren cualquier panel: no hay zonas vacías ni puntos duplicados en
#    ninguna proyección. Eso es lo que "cobertura uniforme" significa.

# %% [markdown]
# ### Diseños clásicos para comparar
#
# - **Box-Behnken:** para ajustar superficies de respuesta cuadráticas
#   (el clásico de optimización de procesos pre-ML)
# - **Plackett-Burman:** para *cribar* muchas variables con poquísimos
#   experimentos (¿cuáles importan?)
# - **Factorial completo 2⁴:** todas las esquinas — crece como 2^K

# %%
X_bb = bbdesign(4, center=3)
X_pb = pbdesign(4)

print("Resumen de diseños para 4 variables:")
print(f"  LHS (elegimos N):      {n_samples} exp — cobertura uniforme, inicializar BO")
print(f"  Box-Behnken:           {X_bb.shape[0]} exp — superficie cuadrática")
print(f"  Plackett-Burman:       {X_pb.shape[0]} exp — screening de variables")
print(f"  Factorial completo:    {2**4} exp — solo esquinas, crece 2^K")

# 🤔 ¿Y con 8 variables? Factorial: 256. LHS: el que tú decidas. Esa es la
#    diferencia que importa cuando cada experimento cuesta $500.

# %% [markdown]
# <div style="background:#FFF3E6;border-left:6px solid #E87C1E;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#C25E08;">🎯 Reto 1 &nbsp;·&nbsp; 5 min</b></div>
#
# **Reto:** tu proceso real tiene **6 variables**. Calcula cuántos
# experimentos exige: (a) factorial completo a 2 niveles, (b) factorial a
# 3 niveles, (c) Box-Behnken, (d) Plackett-Burman. Para (c) y (d) usa pyDOE3
# — qué función y con qué argumentos es investigación suya.
# ¿Cuál elegirías con presupuesto para 50 experimentos, y por qué?

# %%
# [Su código aquí]


# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>P4.2 — Optimización Bayesiana</b> <span style="float:right;background:#E87C1E;color:white;padding:2px 12px;border-radius:12px;font-size:12px;font-weight:600;">⏱ 45 min</span></div>
#
# DoE reparte experimentos *antes de empezar*. BO va más lejos: **usa cada
# resultado para decidir el siguiente experimento**. El loop:
#
# 1. Ajustar un **Gaussian Process** (GP) a los datos disponibles
# 2. El GP da predicción **y** incertidumbre en todo el espacio
# 3. Una **función de adquisición** (EI: *Expected Improvement*) puntúa cada
#    punto candidato balanceando *explotar* (ir a lo que parece bueno) y
#    *explorar* (ir a lo desconocido)
# 4. Evaluar el punto ganador, añadirlo a los datos, repetir
#
# Lo probamos sobre **Hartmann 6D**, una función de prueba estándar con óptimo
# global conocido (3.3224) — así podemos medir qué tan rápido lo encontramos.
#
# **El loop en acción** (versión 1D para verlo): la línea azul es el GP, la
# banda su incertidumbre, la curva naranja es EI (dónde conviene probar), y en
# cada frame se hace "un experimento". Miren cómo la banda se estrecha donde
# ya hay datos y cómo EI persigue el óptimo:
#
# ![Animación del loop de Optimización Bayesiana](media/bo_animacion.gif)

# %%
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.test_functions import Hartmann
from botorch.utils.sampling import draw_sobol_samples

# Usamos LogExpectedImprovement: es el logaritmo de EI. Elige el MISMO punto
# que EI (el log no cambia dónde está el máximo), pero es numéricamente más
# estable cuando EI se vuelve diminuta. botorch lo recomienda sobre EI.

DIM = 6
BOUNDS = torch.stack([torch.zeros(DIM, dtype=torch.double),
                      torch.ones(DIM, dtype=torch.double)])
hartmann = Hartmann(dim=DIM, negate=True)  # negate → problema de MAXIMIZACIÓN

# Inicializar con 10 puntos Sobol (primo del LHS)
torch.manual_seed(1)
train_X = draw_sobol_samples(bounds=BOUNDS, n=10, q=1, seed=1).squeeze(1)
train_Y = hartmann(train_X).unsqueeze(-1)
print(f"Puntos iniciales: {train_X.shape[0]}, mejor valor: {train_Y.max():.3f}")

# %% [markdown]
# <div style="background:#EEF3F9;border-left:6px solid #2E5E9B;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#2E5E9B;">🔍 Cómo funciona este código</b></div>
#
# - botorch trabaja con **tensores de PyTorch** en doble precisión
#   (`torch.double`): `train_X` tiene forma (n, 6) y `train_Y` forma (n, 1).
#   El `.unsqueeze(-1)` agrega esa segunda dimensión que botorch exige.
# - `BOUNDS` es un tensor 2×6: fila 0 = mínimos, fila 1 = máximos de cada
#   variable. Trabajamos siempre en [0, 1] y re-escalamos al final.
# - `draw_sobol_samples` genera puntos cuasi-aleatorios bien repartidos
#   (parecido a LHS). `q=1` significa "un punto por lote".

# %% [markdown]
# **El loop de BO.** Cada iteración = "un experimento". 30 iteraciones tardan
# alrededor de 1 minuto (cada una re-ajusta el GP con todos los datos acumulados):

# %%
from tqdm.auto import tqdm

N_ITER = 30
best_observed_ei = [train_Y.max().item()]

barra = tqdm(range(N_ITER), desc="🧪 Experimentos BO")
for i in barra:
    gp = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    EI = LogExpectedImprovement(model=gp, best_f=train_Y.max())
    candidate, _ = optimize_acqf(EI, bounds=BOUNDS, q=1,
                                 num_restarts=5, raw_samples=64)

    new_y = hartmann(candidate).unsqueeze(-1)
    train_X = torch.cat([train_X, candidate])
    train_Y = torch.cat([train_Y, new_y])
    best_observed_ei.append(train_Y.max().item())
    barra.set_postfix(mejor=f"{best_observed_ei[-1]:.3f}")

# %% [markdown]
# <div style="background:#EEF3F9;border-left:6px solid #2E5E9B;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#2E5E9B;">🔍 Cómo funciona este código</b></div>
#
# - `SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))`: el
#   surrogate. `Standardize` resta la media y divide por la desviación de `Y`
#   (el GP asume datos de media cero y escala 1); las predicciones se
#   des-estandarizan solas.
# - `fit_gpytorch_mll(mll)` ajusta los hiperparámetros del kernel (longitudes
#   de correlación, ruido) maximizando la verosimilitud marginal.
# - `optimize_acqf` busca el máximo de la adquisición: evalúa `raw_samples`
#   puntos al azar, toma los `num_restarts` mejores como arranques y los
#   refina con gradientes (L-BFGS). Devuelve el candidato y su valor.
# - `torch.cat` agrega el nuevo experimento a los datos: el GP de la siguiente
#   iteración ya lo "conoce".

# %% [markdown]
# **La comparación honesta:** ¿y si simplemente hubiéramos tirado dardos?
# Misma cantidad de evaluaciones, elegidas al azar:

# %%
torch.manual_seed(0)
# 5 campañas aleatorias independientes (una sola puede tener mala/buena suerte)
Y_reps = [hartmann(torch.rand(10 + N_ITER, DIM, dtype=torch.double)) for _ in range(5)]
curvas_rnd_h = np.array([[y[:k].max().item() for k in range(10, 10 + N_ITER + 1)]
                         for y in Y_reps])
best_rnd_curve = np.median(curvas_rnd_h, axis=0)

OPTIMUM = 3.3224
plt.figure(figsize=(9, 5))
plt.plot(range(10, 10 + N_ITER + 1), best_observed_ei, label="BO-LogEI (semilla 1)", lw=2, marker="o", ms=3)
plt.plot(range(10, 10 + N_ITER + 1), best_rnd_curve, label="Aleatorio (mediana de 5)", lw=2, ls="--")
plt.fill_between(range(10, 10 + N_ITER + 1), curvas_rnd_h.min(axis=0), curvas_rnd_h.max(axis=0),
                 alpha=0.15, color=NARANJA, label="Aleatorio (rango de 5)")
plt.axhline(OPTIMUM, color="k", ls=":", label=f"Óptimo global = {OPTIMUM}")
plt.xlabel("Número total de evaluaciones")
plt.ylabel("Mejor valor observado")
plt.title("BO-LogEI vs búsqueda aleatoria, Hartmann 6D")
plt.legend()
plt.tight_layout()
plt.show()

frac = 100 * best_observed_ei[-1] / OPTIMUM
print(f"BO alcanzó {frac:.1f}% del óptimo global en {N_ITER} evaluaciones")

# 🤔 ¿En qué evaluación cruza BO el 95% del óptimo? ¿Y el aleatorio?

# %% [markdown]
# **¿Fue suerte?** Una sola campaña no basta para concluir. Repetimos BO con
# otras dos semillas (otros 10 puntos iniciales). Tarda unos 2 minutos.

# %%
def campana_hartmann(semilla, n_init=10, n_iter=N_ITER):
    """Campaña BO completa sobre Hartmann 6D. Devuelve la curva del mejor valor."""
    torch.manual_seed(semilla)
    X = draw_sobol_samples(bounds=BOUNDS, n=n_init, q=1, seed=semilla).squeeze(1)
    Y = hartmann(X).unsqueeze(-1)
    curva = [Y.max().item()]
    for _ in range(n_iter):
        gp = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        acq = LogExpectedImprovement(gp, best_f=Y.max())
        x_new, _ = optimize_acqf(acq, bounds=BOUNDS, q=1, num_restarts=5, raw_samples=64)
        X = torch.cat([X, x_new])
        Y = torch.cat([Y, hartmann(x_new).unsqueeze(-1)])
        curva.append(Y.max().item())
    return curva

curvas_bo = {1: best_observed_ei}
for semilla in [2, 3]:
    curvas_bo[semilla] = campana_hartmann(semilla)

x_eval = range(10, 10 + N_ITER + 1)
plt.figure(figsize=(9, 5))
for semilla, curva in curvas_bo.items():
    plt.plot(x_eval, curva, lw=2, label=f"BO semilla {semilla}")
plt.plot(x_eval, best_rnd_curve, ls="--", color=GRIS, lw=2, label="Aleatorio (mediana de 5)")
plt.axhline(OPTIMUM, color="k", ls=":", label="Óptimo global")
plt.xlabel("Número total de evaluaciones")
plt.ylabel("Mejor valor observado")
plt.title("BO con 3 semillas: el resultado final varía")
plt.legend()
plt.tight_layout()
plt.show()

for semilla, curva in curvas_bo.items():
    print(f"Semilla {semilla}: final {curva[-1]:.2f} ({100*curva[-1]/OPTIMUM:.0f}% del óptimo)")

# 🤔 BO le gana a la mediana aleatoria, pero alguna semilla puede quedar
#    atrapada en un máximo local (Hartmann 6D tiene varios). En un laboratorio
#    esto se traduce en: una campaña no es una prueba; conviene repetir o
#    explorar más (UCB con beta alto, más puntos iniciales).

# %% [markdown]
# <div style="background:#FFF3E6;border-left:6px solid #E87C1E;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#C25E08;">🎯 Reto 2 &nbsp;·&nbsp; 10 min</b></div>
#
# **Reto:** en el laboratorio real, cada punto inicial también
# cuesta. Repite el loop de BO arrancando con solo **3 puntos Sobol** en lugar
# de 10 (mismo presupuesto total de 40 evaluaciones: 3 + 37 iteraciones).
#
# - ¿Converge igual? ¿Más lento? ¿Se atasca?
# - Corre tu versión 2–3 veces cambiando la semilla. ¿Qué observas sobre la
#   *variabilidad* del resultado? ¿Qué le recomendarías a alguien que solo
#   puede pagar 3 experimentos iniciales?

# %%
# [Su código aquí. Pista: la función campana_hartmann tiene el argumento n_init]


# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>P4.3 — BO sobre un proceso PVD "real"</b> <span style="float:right;background:#E87C1E;color:white;padding:2px 12px;border-radius:12px;font-size:12px;font-weight:600;">⏱ 25 min</span></div>
#
# Ahora el caso de manufactura: optimizar la **dureza** de un recubrimiento
# (Ti,Al)N ajustando 4 variables de proceso.
#
# **El truco didáctico:** tenemos un dataset histórico de 200 experimentos
# (`superhard_experimental.csv`). Construimos con él un **oráculo**
# (interpolador RBF) que "juega a ser el laboratorio": cada consulta al
# oráculo simula hacer un experimento de 8 horas. Así podemos correr la
# campaña completa en clase.

# %%
from scipy.interpolate import RBFInterpolator

df_pvd = pd.read_csv(DATOS / "superhard_experimental.csv")
print(f"Dataset histórico: {df_pvd.shape}")
print(df_pvd.columns.tolist())
df_pvd.describe().round(2).loc[["min", "mean", "max"]]

# %%
feature_cols_pvd = ["T_proceso", "P_N2", "bias_voltaje", "flujo_Ti_Al"]
target_col = "dureza_HV"

X_pvd = df_pvd[feature_cols_pvd].values
y_pvd = df_pvd[target_col].values

# Normalizar el espacio a [0,1]^4 ANTES de construir el oráculo.
# ⚠ Si no, la distancia RBF queda dominada por la variable de mayor escala
#   (aquí T ~ cientos vs P ~ 1) y el interpolador produce artefactos.
X_min, X_max = X_pvd.min(axis=0), X_pvd.max(axis=0)
X_norm = (X_pvd - X_min) / (X_max - X_min)

# El "laboratorio simulado" — trabaja en coordenadas normalizadas
oracle = RBFInterpolator(X_norm, y_pvd, kernel="thin_plate_spline")

BOUNDS_PVD = torch.stack([torch.zeros(4, dtype=torch.double),
                          torch.ones(4, dtype=torch.double)])

def oracle_torch(X_t: torch.Tensor) -> torch.Tensor:
    """Simula el experimento: tensor normalizado → dureza HV."""
    return torch.tensor(oracle(X_t.detach().numpy()), dtype=torch.double).unsqueeze(-1)

# %% [markdown]
# <div style="background:#EEF3F9;border-left:6px solid #2E5E9B;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#2E5E9B;">🔍 Cómo funciona este código</b></div>
#
# - `RBFInterpolator(X_norm, y_pvd, kernel="thin_plate_spline")` construye una
#   superficie suave que pasa **exactamente** por los 200 datos históricos y
#   rellena entre ellos. Es nuestro "laboratorio de mentira".
# - Ojo: el oráculo no tiene ruido y puede extrapolar mal fuera de la nube de
#   datos. Un laboratorio real sí tiene ruido; por eso el GP estima un término
#   de ruido al ajustarse.
# - `oracle_torch` es un adaptador: botorch habla en tensores de PyTorch y
#   scipy en arreglos de NumPy. `.detach().numpy()` convierte en un sentido y
#   `torch.tensor(...)` en el otro.

# %% [markdown]
# **Escenario realista:** nuestro proceso histórico *no estaba optimizado*.
# Arrancamos la campaña desde 5 mediciones de la **mitad menos dura** del
# histórico (como quien hereda un proceso mediocre) y damos a BO un
# presupuesto de **20 experimentos nuevos**.

# %%
rng = np.random.default_rng(42)
torch.manual_seed(42)

mitad_peor = np.argsort(y_pvd)[: len(y_pvd) // 2]
idx_init = rng.choice(mitad_peor, 5, replace=False)
train_X_pvd = torch.tensor(X_norm[idx_init], dtype=torch.double)
train_Y_pvd = torch.tensor(y_pvd[idx_init], dtype=torch.double).unsqueeze(-1)

print(f"Punto de partida (mejor de los 5 iniciales): {train_Y_pvd.max():.0f} HV")

best_bo = [train_Y_pvd.max().item()]

barra = tqdm(range(20), desc="🏭 Campaña PVD")
for i in barra:
    gp = SingleTaskGP(train_X_pvd, train_Y_pvd, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    EI = LogExpectedImprovement(gp, best_f=train_Y_pvd.max())
    cand, _ = optimize_acqf(EI, bounds=BOUNDS_PVD, q=1,
                            num_restarts=5, raw_samples=32)
    new_y = oracle_torch(cand)
    train_X_pvd = torch.cat([train_X_pvd, cand])
    train_Y_pvd = torch.cat([train_Y_pvd, new_y])
    best_bo.append(train_Y_pvd.max().item())
    barra.set_postfix(HV=f"{best_bo[-1]:.0f}")

# %% [markdown]
# **La comparación honesta.** Una sola corrida aleatoria puede tener suerte —
# por eso el competidor justo es la **distribución** de la estrategia aleatoria:
# la repetimos 10 veces (es gratis: solo consultas al oráculo).

# %%
curvas_rnd = []
for rep in range(10):
    rng_r = np.random.default_rng(100 + rep)
    idx0 = rng_r.choice(mitad_peor, 5, replace=False)
    best = y_pvd[idx0].max()
    curva = [best]
    for _ in range(20):
        val = float(oracle(rng_r.random((1, 4)))[0])
        best = max(best, val)
        curva.append(best)
    curvas_rnd.append(curva)
curvas_rnd = np.array(curvas_rnd)

best_global = y_pvd.max()
rnd_final = curvas_rnd[:, -1]
print(f"HV máximo del dataset histórico:  {best_global:.0f}")
print(f"BO tras 20 experimentos:          {best_bo[-1]:.0f}  ({100*best_bo[-1]/best_global:.1f}% del máximo)")
print(f"Aleatorio (10 repeticiones):      mediana {np.median(rnd_final):.0f}, "
      f"rango [{rnd_final.min():.0f}–{rnd_final.max():.0f}]")
print(f"Repeticiones aleatorias que alcanzan el 98% del máximo: "
      f"{(rnd_final >= 0.98 * best_global).sum()}/10")

# %%
x_exp = np.arange(len(best_bo))
try:
    if not USAR_PLOTLY:
        raise ImportError("plotly desactivado")
    import plotly.graph_objects as go

    figx = go.Figure()
    figx.add_trace(go.Scatter(  # banda del aleatorio (rango de 10 reps)
        x=np.concatenate([x_exp, x_exp[::-1]]),
        y=np.concatenate([curvas_rnd.max(axis=0), curvas_rnd.min(axis=0)[::-1]]),
        fill="toself", fillcolor="rgba(217,111,15,0.15)",
        line=dict(width=0), name="Aleatorio (rango de 10)", hoverinfo="skip"))
    figx.add_trace(go.Scatter(
        x=x_exp, y=np.median(curvas_rnd, axis=0), name="Aleatorio (mediana de 10)",
        line=dict(color=NARANJA, dash="dash", width=2),
        hovertemplate="exp %{x}: %{y:.0f} HV<extra>Aleatorio</extra>"))
    figx.add_trace(go.Scatter(
        x=x_exp, y=best_bo, name="BO-LogEI (1 campaña)", mode="lines+markers",
        line=dict(color=AZUL, width=2.5), marker=dict(size=6),
        hovertemplate="exp %{x}: %{y:.0f} HV<extra>BO-LogEI</extra>"))
    figx.add_hline(y=best_global, line_dash="dot", line_color=TINTA,
                   annotation_text=f"máximo histórico = {best_global:.0f} HV")
    figx.add_hline(y=0.98 * best_global, line_dash="dot", line_color=GRIS,
                   annotation_text="98% del máximo")
    figx.update_layout(
        title=dict(text="Campaña de optimización de dureza (Ti,Al)N", font_weight=700),
        xaxis_title="Experimento nuevo", yaxis_title="Mejor dureza encontrada (HV)",
        width=850, height=520, font_color=TINTA, plot_bgcolor="white",
        hovermode="x unified", legend=dict(orientation="h", y=-0.2))
    figx.update_xaxes(gridcolor="#E8EAEE")
    figx.update_yaxes(gridcolor="#E8EAEE")
    figx.show()
except ImportError:
    plt.figure(figsize=(9, 5))
    plt.plot(x_exp, best_bo, label="BO-LogEI (1 campaña)", lw=2, marker="o", ms=4)
    plt.plot(x_exp, np.median(curvas_rnd, axis=0), label="Aleatorio (mediana de 10)", lw=2, ls="--", color=NARANJA)
    plt.fill_between(x_exp, curvas_rnd.min(axis=0), curvas_rnd.max(axis=0),
                     alpha=0.2, color=NARANJA, label="Aleatorio (rango de 10)")
    plt.axhline(best_global, color=TINTA, ls=":", label=f"Máximo histórico = {best_global:.0f} HV")
    plt.axhline(0.98 * best_global, color=GRIS, ls=":", lw=1, label="98% del máximo")
    plt.xlabel("Experimento nuevo")
    plt.ylabel("Mejor dureza encontrada (HV)")
    plt.title("Campaña de optimización de dureza (Ti,Al)N — inicio no optimizado")
    plt.legend()
    plt.tight_layout()
    plt.show()

# 🤔 Los ÚLTIMOS puntos de dureza son los caros: la región con HV ≥ 98% del
#    máximo ocupa <1% del espacio de proceso. Ahí es donde el azar falla
#    (~1-2 de cada 10 campañas lo logran) y BO llega sistemáticamente.

# %% [markdown]
# **¿Y cuáles son las condiciones de proceso ganadoras?** Esto es lo que uno
# se lleva al laboratorio:

# %%
best_idx = int(train_Y_pvd.argmax())
best_X_real = train_X_pvd[best_idx].numpy() * (X_max - X_min) + X_min

print("Mejores condiciones encontradas por BO:")
for col, val in zip(feature_cols_pvd, best_X_real):
    print(f"  {col:15s}: {val:8.2f}")
print(f"  {'dureza_HV':15s}: {train_Y_pvd[best_idx].item():8.0f}")

# %% [markdown]
# ### 🔬 Explorador interactivo del proceso (opcional)
#
# Mueve los sliders de `bias` y `flujo` y mira cómo cambia el mapa de dureza
# en el plano T–P según el oráculo. ¿Encuentras a ojo la zona que BO encontró
# sola?
#
# > ⚠️ Los sliders funcionan en **JupyterLab**; en VS Code pueden no renderizar
# > (si no ves nada, sáltate esta celda — es un extra).

# %%
try:
    from ipywidgets import interact, FloatSlider

    _T = np.linspace(0, 1, 60)
    _P = np.linspace(0, 1, 60)
    _TT, _PP = np.meshgrid(_T, _P)

    def mapa_dureza(bias_norm=0.5, flujo_norm=0.5):
        pts = np.column_stack([_TT.ravel(), _PP.ravel(),
                               np.full(_TT.size, bias_norm), np.full(_TT.size, flujo_norm)])
        Z = oracle(pts).reshape(_TT.shape)
        plt.figure(figsize=(7, 5))
        ext = [X_min[0], X_max[0], X_min[1], X_max[1]]
        plt.imshow(Z, origin="lower", aspect="auto", extent=ext, cmap="viridis")
        plt.colorbar(label="dureza_HV (oráculo)")
        bias_real = bias_norm * (X_max[2] - X_min[2]) + X_min[2]
        flujo_real = flujo_norm * (X_max[3] - X_min[3]) + X_min[3]
        plt.xlabel("T_proceso (°C)")
        plt.ylabel("P_N2 (Pa)")
        plt.title(f"Dureza en el plano T–P  (bias={bias_real:.0f} V, flujo={flujo_real:.2f})")
        plt.show()

    interact(mapa_dureza,
             bias_norm=FloatSlider(0.5, min=0, max=1, step=0.05, description="bias"),
             flujo_norm=FloatSlider(0.5, min=0, max=1, step=0.05, description="flujo"))
except Exception as e:
    print(f"ipywidgets no disponible aquí ({type(e).__name__}) — celda opcional, continuar sin ella.")

# %% [markdown]
# <div style="background:#FFF3E6;border-left:6px solid #E87C1E;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#C25E08;">🎯 Reto 3 &nbsp;·&nbsp; opcional / para casa</b></div>
#
# **Reto:** el cliente acaba de añadir un requisito: además de
# dureza alta, el recubrimiento necesita **CoF < 0.4** (el dataset tiene la
# columna `CoF`). Tu campaña actual lo ignora por completo.
#
# Propón e implementa UNA forma de incorporarlo. Rutas posibles (investiga y
# elige): penalizar el objetivo, construir un segundo oráculo para CoF y
# filtrar candidatos, o buscar en la documentación de botorch qué existe para
# *constrained optimization*. No hay respuesta única — defiende la tuya.

# %%
# [Su código aquí]

# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>P4.4 — Discusión en grupo</b> <span style="float:right;background:#E87C1E;color:white;padding:2px 12px;border-radius:12px;font-size:12px;font-weight:600;">⏱ 5 min</span></div>
#
# 1. ¿Cuántos experimentos ahorró BO frente al aleatorio en *tu* corrida?
# 2. Si cada experimento PVD tarda **8 horas y cuesta \$500**, ¿cuánto
#    representa ese ahorro en tu laboratorio?
# 3. ¿Cuándo preferirías un Box-Behnken clásico en lugar de BO?
#    (Pista: ¿qué pasa si necesitas *entender* el proceso, no solo optimizarlo?
#    ¿Y si puedes correr los 25 experimentos en paralelo?)
#
# ---
# ## 🏠 Ejercicio del Día 4 (para casa)
#
# Repite la campaña PVD usando **UCB con β = 2.0** en lugar de EI
# (`UpperConfidenceBound(gp, beta=2.0)`, ya está importado):
#
# 1. ¿En qué iteración alcanza cada método el 98% y el 99% del HV máximo?
#    (el 95% lo alcanzan ambos casi de inmediato: no discrimina).
#    Grafica ambas curvas de convergencia juntas.
# 2. Scatter de todos los puntos evaluados (color = número de iteración) en el
#    plano `T_proceso` vs `P_N2`. ¿Cómo difiere la *exploración* de EI vs UCB?
#
# ---
# ### 🔜 Conexión con el Día 5
# > Mañana no hay contenido nuevo: **ponen todo junto.** Un brief industrial,
# > datos reales, el pipeline que ustedes elijan — y a presentar resultados.
