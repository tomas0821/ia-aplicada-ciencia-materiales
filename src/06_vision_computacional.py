# %% [markdown]
# <div style="background:linear-gradient(90deg,#1A2E4A 0%,#2E5E9B 100%);color:white;padding:20px 26px;border-radius:12px;">
# <span style="font-size:12px;letter-spacing:3px;color:#D6E8F7;">CURSO IA APLICADA A CIENCIA DE MATERIALES &nbsp;·&nbsp; DÍA 3 · VISIÓN COMPUTACIONAL</span><br>
# <span style="font-size:26px;font-weight:700;">🔬 Redes que "leen" difractogramas e imágenes SEM</span><br>
# <span style="font-size:14px;color:#D6E8F7;">Modelos ya entrenados: hoy los usamos, los evaluamos y buscamos dónde fallan</span>
# </div>
#
# | Bloque | Contenido | Tiempo |
# |--------|-----------|--------|
# | P3.5a | XRD: sistema cristalino con una red convolucional 1D | 22 min |
# | P3.5b | SEM: clasificar imágenes de microscopía con ResNet-18 | 13 min |
# | | Cierre y discusión | 5 min |
#
# **La idea central:** una red convolucional aprende sola qué patrones mirar
# (picos, texturas, bordes). Entrenarla cuesta horas de GPU y miles de ejemplos,
# así que **los modelos ya vienen entrenados** (se entrenaron en Kabré antes del
# curso). Hoy hacemos lo que hace un usuario responsable de un modelo ajeno:
# medir qué tan bien funciona, entender qué mira y encontrar dónde se equivoca.
#
# **Datos (todos públicos, licencia CC BY 4.0):**
# - 52 809 difractogramas **simulados** con pymatgen a partir de estructuras de
#   Materials Project (conjunto `matbench_mp_e_form`).
# - 328 difractogramas **experimentales** de la base opXRD (Hollarek et al. 2025).
# - Imágenes SEM de la base NFFA-Europe (Aversa et al. 2018), 10 categorías.
#
# > Este notebook corre en CPU (kura) o GPU (nukwa). No necesita internet.

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

# %%
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay


def encontrar_datos() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents[:3]]:
        if (base / "datos").is_dir():
            return base / "datos"
    raise FileNotFoundError("No encuentro la carpeta datos/")


DATOS = encontrar_datos()
VISION = DATOS / "vision"
torch.set_grad_enabled(False)     # solo vamos a predecir; los gradientes se activan donde hagan falta
print("Archivos:", sorted(p.name for p in VISION.iterdir()))

# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>P3.5a — XRD: ¿qué sistema cristalino es?</b> <span style="float:right;background:#E87C1E;color:white;padding:2px 12px;border-radius:12px;font-size:12px;font-weight:600;">⏱ 22 min</span></div>
#
# Un difractograma de polvo es una **imagen de una dimensión**: intensidad en
# función de 2θ. La simetría del cristal decide cuántos picos hay y dónde: un
# cúbico tiene pocos picos muy ordenados; un triclínico, muchos y sin patrón
# evidente. ¿Puede una red aprender esa relación solo viendo ejemplos?
#
# Primero, los datos de prueba simulados (el modelo **no** los vio al entrenar):

# %%
sim = np.load(VISION / "xrd_simulados_prueba.npz", allow_pickle=True)
x2t = sim["x"]                                   # 1200 valores de 2θ entre 10 y 70°
X_sim = sim["perfiles"].astype(np.float32)      # (n, 1200), intensidad normalizada a 1
y_sim = sim["y"]
SISTEMAS = [str(s) for s in sim["sistemas"]]
NOMBRES = ["triclínico", "monoclínico", "ortorrómbico", "tetragonal", "trigonal", "hexagonal", "cúbico"]
print(f"{len(y_sim)} patrones simulados de prueba, {len(SISTEMAS)} sistemas cristalinos")

fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
for ax, k in zip(axes, [6, 3, 0]):
    i = np.where(y_sim == k)[0][0]
    ax.plot(x2t, X_sim[i], color=[AZUL, VERDE, NARANJA][[6, 3, 0].index(k)], lw=1.2)
    ax.set_title(f"{sim['formula'][i]}  ·  {NOMBRES[k]} (grupo {sim['sg'][i]})", fontsize=11, loc="left")
    ax.set_ylabel("I (norm.)")
axes[-1].set_xlabel("2θ (grados, Cu Kα)")
plt.tight_layout(); plt.show()

# 🤔 El cúbico tiene pocos picos y bien separados; el triclínico, muchos.
#    Los patrones incluyen ruido, fondo y picos anchos a propósito: así se
#    entrenó el modelo, para parecerse a un difractómetro real.

# %% [markdown]
# **El modelo.** Una red convolucional 1D: cinco bloques que "barren" el
# difractograma con filtros pequeños, y dos capas finales que deciden la clase.
# La definición tiene que ser **idéntica** a la usada al entrenar; los pesos
# (`xrd_cnn1d.pt`, 1.3 MB) son los números que la red aprendió.

# %%
class CNN1D(nn.Module):
    """Cinco bloques convolución + normalización + ReLU + submuestreo, y dos capas finales."""
    def __init__(self, n_clases=7):
        super().__init__()
        capas, cin = [], 1
        for cout, k in [(32, 15), (64, 9), (128, 7), (128, 5), (256, 3)]:
            capas += [nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.ReLU(), nn.MaxPool1d(2)]
            cin = cout
        self.features = nn.Sequential(*capas)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * cin, 128), nn.ReLU(), nn.Linear(128, n_clases))

    def forward(self, x):                       # x: (lote, 1200)
        h = self.features(torch.sqrt(x.clamp_min(0))[:, None, :])   # raíz: realza picos débiles
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)
        return self.head(h)


modelo_xrd = CNN1D(len(SISTEMAS))
modelo_xrd.load_state_dict(torch.load(VISION / "modelos" / "xrd_cnn1d.pt", map_location="cpu"))
modelo_xrd.eval()
print(f"Parámetros entrenables: {sum(p.numel() for p in modelo_xrd.parameters()):,}")

# %% [markdown]
# <div style="background:#EEF3F9;border-left:6px solid #2E5E9B;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#2E5E9B;">🔍 Cómo funciona este código</b></div>
#
# - `nn.Conv1d(cin, cout, k)`: `cout` filtros de ancho `k` que se deslizan por el
#   patrón. Cada filtro aprende a reconocer una forma local: un pico, un doblete,
#   un hombro. Las capas siguientes combinan esas formas en patrones más grandes.
# - `nn.MaxPool1d(2)`: se queda con el máximo de cada par de puntos. Reduce la
#   resolución a la mitad y hace al modelo tolerante a pequeños corrimientos.
# - `torch.sqrt(...)`: la raíz de la intensidad realza los picos débiles, que
#   también llevan información de simetría.
# - `h.mean(-1)` y `h.amax(-1)`: resumen cada filtro en dos números (promedio y
#   máximo a lo largo de 2θ), sin importar en qué ángulo apareció el patrón.
# - `load_state_dict(torch.load(...))`: copia los pesos aprendidos en la red.
#   `model.eval()` apaga el *dropout* y fija la normalización para predecir.

# %%
def predecir_xrd(X, lote=512):
    """Devuelve probabilidades (n, 7) para una matriz de patrones (n, 1200)."""
    salidas = []
    for k in range(0, len(X), lote):
        salidas.append(torch.softmax(modelo_xrd(torch.tensor(X[k:k + lote])), 1))
    return torch.cat(salidas).numpy()


prob_sim = predecir_xrd(X_sim)
pred_sim = prob_sim.argmax(1)
print(f"Exactitud en simulados de prueba: {accuracy_score(y_sim, pred_sim):.1%}  (azar: {1/7:.1%})")

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(y_sim, pred_sim, display_labels=NOMBRES, normalize="true",
                                        values_format=".2f", cmap="Blues", ax=ax, colorbar=False)
plt.xticks(rotation=45, ha="right"); ax.set_title("XRD simulados: fila = real, columna = predicho")
plt.tight_layout(); plt.show()

# %% [markdown]
# 🤔 Lean la matriz: ¿qué sistemas reconoce mejor? ¿Con cuáles se confunde
# el modelo? Pista física: tetragonal y cúbico se parecen si c ≈ a; trigonal y
# hexagonal comparten muchas reflexiones; monoclínico y triclínico producen
# "bosques" de picos parecidos.

# %% [markdown]
# **¿Qué mira el modelo?** El *mapa de saliencia* responde: si muevo un poco la
# intensidad en este ángulo, ¿cuánto cambia la decisión? Las zonas altas son las
# que el modelo está usando.

# %%
def saliencia(x):
    """Gradiente de la clase predicha respecto a la entrada (valor absoluto)."""
    with torch.enable_grad():
        xt = torch.tensor(np.clip(x, 1e-4, None)[None], requires_grad=True)   # evita la derivada infinita de la raíz en 0
        salida = modelo_xrd(xt)
        clase = int(salida.argmax())
        salida[0, clase].backward()
    return xt.grad[0].abs().numpy(), clase


i = int(np.where((y_sim == 6) & (pred_sim == 6))[0][0])      # un cúbico bien clasificado
s, c = saliencia(X_sim[i])
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.plot(x2t, X_sim[i], color=GRIS, lw=1, label="patrón")
ax.fill_between(x2t, 0, s / s.max(), color=NARANJA, alpha=0.5, label="saliencia")
ax.set_title(f"{sim['formula'][i]}: predicho {NOMBRES[c]}"); ax.set_xlabel("2θ (grados)"); ax.legend()
plt.tight_layout(); plt.show()

# 🤔 ¿La saliencia se concentra en los picos, en el fondo o en el ruido?
#    Un modelo que mira el ruido es una alerta: aprendió un atajo.

# %% [markdown]
# **La prueba de fuego: datos experimentales.** El modelo solo vio patrones
# simulados. Ahora le mostramos 328 difractogramas medidos en laboratorio (opXRD,
# contribución del CNRS), con una sola fase y sistema cristalino conocido.

# %%
real = np.load(VISION / "xrd_reales_opxrd.npz", allow_pickle=True)
X_real = real["perfiles"].astype(np.float32)
y_real = real["y"]
comp = real["composicion"]
elementos = [{t.rstrip("0123456789.") for t in c.split()} for c in comp]    # "H84 C24 N8" -> {"H","C","N"}
organico = np.array([{"C", "H"} <= e for e in elementos])

pred_real = predecir_xrd(X_real).argmax(1)
mayoritaria = np.bincount(y_real).argmax()
print(f"Exactitud en patrones reales:        {accuracy_score(y_real, pred_real):.1%}")
print(f"  inorgánicos ({(~organico).sum()}):              {accuracy_score(y_real[~organico], pred_real[~organico]):.1%}")
print(f"  moleculares/orgánicos ({organico.sum()}):   {accuracy_score(y_real[organico], pred_real[organico]):.1%}")
print(f"Siempre decir '{NOMBRES[mayoritaria]}':  {(y_real == mayoritaria).mean():.1%}")

fig, ax = plt.subplots(figsize=(10, 3.4))
i = int(np.where(~organico)[0][0])
ax.plot(x2t, X_real[i], color=AZUL, lw=1)
ax.set_title(f"Patrón experimental: {comp[i]} ({NOMBRES[y_real[i]]}); el modelo dice {NOMBRES[pred_real[i]]}", fontsize=11)
ax.set_xlabel("2θ (grados)"); plt.tight_layout(); plt.show()

# %% [markdown]
# <div style="background:#EEF3F9;border-left:6px solid #2E5E9B;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#2E5E9B;">🔍 Cómo funciona este código</b></div>
#
# - Los patrones reales se interpolaron a la **misma rejilla** del modelo (1200
#   puntos entre 10 y 70°) y se normalizaron a máximo 1. Sin ese paso el modelo
#   recibiría números que no entiende.
# - `organico` marca las fórmulas con C e H: cristales moleculares, que casi no
#   existen en Materials Project. Separarlos muestra cuánto del error viene de
#   pedirle al modelo algo que nunca vio.
# - La línea "siempre decir..." es la **referencia trivial**: un modelo que no la
#   supera no aporta nada.

# %% [markdown]
# 🤔 **Discusión (2 min):** ¿por qué cae tanto la exactitud? Piensen en lo que
# el simulador no incluye: orientación preferencial, fases secundarias, tamaño
# de cristalito, radiación Kα2, portamuestras, y química que no está en el
# entrenamiento. A esto se le llama **cambio de dominio** (*domain shift*) y es
# el problema número uno al pasar de la simulación al laboratorio.

# %% [markdown]
# **¿Y si le mostramos algunos patrones reales?** En un laboratorio normalmente
# hay unas decenas de difractogramas propios ya identificados. Usemos la mitad
# de los patrones reales para **reajustar solo las dos capas finales** del modelo
# (las convoluciones quedan congeladas) y midamos en la otra mitad. Separamos
# por composición: un mismo compuesto no puede quedar en los dos lados.

# %%
import copy
from sklearn.model_selection import GroupShuffleSplit

tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=0).split(X_real, y_real, groups=comp))
ajustado = copy.deepcopy(modelo_xrd)
for p in ajustado.features.parameters():
    p.requires_grad = False                     # congelar los filtros aprendidos con datos simulados
opt = torch.optim.Adam(ajustado.head.parameters(), lr=1e-3)
Xtr, ytr = torch.tensor(X_real[tr]), torch.tensor(y_real[tr])
with torch.enable_grad():
    for epoca in range(60):
        ajustado.train(); ajustado.features.eval()
        opt.zero_grad()
        perdida = nn.functional.cross_entropy(ajustado(Xtr), ytr)
        perdida.backward(); opt.step()
ajustado.eval()

antes = modelo_xrd(torch.tensor(X_real[te])).argmax(1).numpy()
despues = ajustado(torch.tensor(X_real[te])).argmax(1).numpy()
print(f"Patrones reales para ajustar: {len(tr)}  ·  para evaluar: {len(te)}")
print(f"Exactitud en la mitad de prueba, antes del ajuste:   {accuracy_score(y_real[te], antes):.1%}")
print(f"Exactitud en la mitad de prueba, después del ajuste: {accuracy_score(y_real[te], despues):.1%}")

# %% [markdown]
# <div style="background:#EEF3F9;border-left:6px solid #2E5E9B;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#2E5E9B;">🔍 Cómo funciona este código</b></div>
#
# - `copy.deepcopy`: trabajamos sobre una copia; el modelo original queda intacto.
# - `requires_grad = False` congela los filtros convolucionales: solo cambian las
#   dos capas finales (unos 67 000 de los 325 000 pesos). Con pocos datos eso evita
#   que el modelo "memorice" los patrones de ajuste.
# - `GroupShuffleSplit(groups=comp)`: si un compuesto aparece dos veces, queda
#   entero de un solo lado; si no, estaríamos evaluando con patrones casi vistos.
# - El ciclo de 60 épocas es el mismo esquema del Día 4 con botorch: calcular la
#   pérdida, `backward()` para los gradientes y `opt.step()` para actualizar.
#
# 🤔 Con unos 160 patrones propios el modelo mejora muchísimo. Esa es la receta
# práctica: **entrenar con simulados + ajustar con pocos datos reales**.

# %% [markdown]
# <div style="background:#FFF3E6;border-left:6px solid #E87C1E;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#C25E08;">🎯 Reto A &nbsp;·&nbsp; 6 min</b></div>
#
# **Reto:** ¿qué tan robusto es el modelo? Usen la función `perturbar` para
# desplazar los patrones simulados en 2θ (error de altura de muestra) o
# agregarles ruido, y midan la exactitud para varios valores. Grafiquen
# exactitud contra desplazamiento (por ejemplo de 0 a 1°). ¿A partir de qué
# desplazamiento el modelo deja de servir?

# %%
def perturbar(X, desplazamiento=0.0, ruido=0.0, semilla=0):
    """Desplaza los patrones `desplazamiento` grados en 2θ y agrega ruido gaussiano."""
    rng = np.random.default_rng(semilla)
    paso = x2t[1] - x2t[0]
    n = int(round(desplazamiento / paso))
    Xp = np.roll(X, n, axis=1)
    if n > 0:
        Xp[:, :n] = X[:, :1]
    elif n < 0:
        Xp[:, n:] = X[:, -1:]
    Xp = Xp + rng.normal(0, ruido, X.shape)
    Xp = np.clip(Xp, 0, None)
    return (Xp / Xp.max(1, keepdims=True)).astype(np.float32)


# [Su código aquí]

# %% [markdown]
# ---
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>P3.5b — SEM: ¿qué muestra esta imagen?</b> <span style="float:right;background:#E87C1E;color:white;padding:2px 12px;border-radius:12px;font-size:12px;font-weight:600;">⏱ 13 min</span></div>
#
# Ahora imágenes de verdad: micrografías SEM de la base NFFA-Europe, en 10
# categorías (partículas, nanohilos, fibras, películas, MEMS, etc.). El modelo es
# una **ResNet-18**: una red que aprendió a ver con 1.2 millones de fotos
# cotidianas (ImageNet) y que luego ajustamos con ~3500 imágenes SEM.
# Eso se llama **aprendizaje por transferencia**.

# %%
from PIL import Image
import torchvision
from torchvision import transforms as T

info_sem = json.load(open(VISION / "modelos" / "sem_info.json"))
CLASES_SEM = info_sem["clases"]
prep = T.Compose([T.Grayscale(3), T.CenterCrop(224), T.ToTensor(), T.Normalize(info_sem["mean"], info_sem["std"])])

archivos, y_sem = [], []
for k, c in enumerate(CLASES_SEM):
    fs = sorted((VISION / "sem_prueba" / c).glob("*.png"))
    archivos += fs; y_sem += [k] * len(fs)
y_sem = np.array(y_sem)
print(f"{len(archivos)} imágenes de prueba en {len(CLASES_SEM)} clases")

fig, axes = plt.subplots(2, 5, figsize=(13, 5.6))
for ax, c in zip(axes.ravel(), CLASES_SEM):
    f = sorted((VISION / "sem_prueba" / c).glob("*.png"))[0]
    ax.imshow(Image.open(f), cmap="gray"); ax.set_title(c.replace("_", " "), fontsize=10); ax.axis("off")
plt.tight_layout(); plt.show()

# %%
modelo_sem = torchvision.models.resnet18(weights=None)
modelo_sem.fc = nn.Linear(512, len(CLASES_SEM))
pesos = torch.load(VISION / "modelos" / "sem_resnet18.pt", map_location="cpu")
modelo_sem.load_state_dict({k: v.float() for k, v in pesos.items()})
modelo_sem.eval()

X_img = torch.stack([prep(Image.open(f)) for f in archivos])
prob_sem = torch.cat([torch.softmax(modelo_sem(X_img[k:k + 50]), 1) for k in range(0, len(X_img), 50)]).numpy()
pred_sem = prob_sem.argmax(1)
print(f"Exactitud SEM: {accuracy_score(y_sem, pred_sem):.1%}  (azar: {1/len(CLASES_SEM):.0%})")

fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay.from_predictions(y_sem, pred_sem, display_labels=[c.replace("_", " ") for c in CLASES_SEM],
                                        cmap="Blues", ax=ax, colorbar=False)
plt.xticks(rotation=45, ha="right"); ax.set_title("SEM: fila = real, columna = predicho")
plt.tight_layout(); plt.show()

# %% [markdown]
# <div style="background:#EEF3F9;border-left:6px solid #2E5E9B;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#2E5E9B;">🔍 Cómo funciona este código</b></div>
#
# - `resnet18(weights=None)` crea la arquitectura vacía; `modelo_sem.fc` cambia
#   la última capa: de 1000 clases de ImageNet a nuestras 10 categorías.
# - Los pesos se guardaron en media precisión (`float16`) para que el archivo
#   pese la mitad; `v.float()` los convierte de vuelta.
# - `prep`: la imagen gris se copia en 3 canales (la red espera color), se
#   recorta el centro de 224 × 224 píxeles y se normaliza con las medias y
#   desviaciones de ImageNet, exactamente como en el entrenamiento.
# - `torch.stack` junta las 200 imágenes en un solo tensor (200, 3, 224, 224);
#   las pasamos de 50 en 50 para no llenar la memoria.

# %% [markdown]
# **Los errores enseñan más que los aciertos.** Veamos las imágenes mal
# clasificadas: ¿se equivoca el modelo o la etiqueta es discutible?

# %%
malas = np.where(pred_sem != y_sem)[0]
print(f"{len(malas)} errores de {len(y_sem)}")
fig, axes = plt.subplots(1, min(6, len(malas)), figsize=(15, 3.2), squeeze=False)
for ax, i in zip(axes[0], malas[:6]):
    ax.imshow(Image.open(archivos[i]), cmap="gray"); ax.axis("off")
    ax.set_title(f"real: {CLASES_SEM[y_sem[i]]}\npred: {CLASES_SEM[pred_sem[i]]} ({prob_sem[i].max():.0%})", fontsize=9)
plt.tight_layout(); plt.show()

# %% [markdown]
# **¿Cómo "ve" el modelo las imágenes?** Quitamos la última capa y miramos el
# vector de 512 números que la red calcula para cada imagen. Lo proyectamos a
# 2D con PCA: si las categorías forman grupos, la red aprendió a separarlas.

# %%
from sklearn.decomposition import PCA

cabeza = modelo_sem.fc
modelo_sem.fc = nn.Identity()                   # ahora la red devuelve el vector de 512
emb = torch.cat([modelo_sem(X_img[k:k + 50]) for k in range(0, len(X_img), 50)]).numpy()
modelo_sem.fc = cabeza                          # la dejamos como estaba

z = PCA(2).fit_transform(emb)
colores = plt.cm.tab10(np.arange(len(CLASES_SEM)))
fig, ax = plt.subplots(figsize=(8, 6))
for k, c in enumerate(CLASES_SEM):
    ax.scatter(*z[y_sem == k].T, s=22, color=colores[k], label=c.replace("_", " "))
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Lo que ve la red: 512 números por imagen, proyectados a 2D")
ax.legend(fontsize=8, ncol=2, loc="best"); plt.tight_layout(); plt.show()

# %% [markdown]
# <div style="background:#FFF3E6;border-left:6px solid #E87C1E;padding:8px 14px;border-radius:6px;margin-top:8px;"><b style="color:#C25E08;">🎯 Reto B &nbsp;·&nbsp; 5 min</b></div>
#
# **Reto:** la probabilidad máxima (`prob_sem.max(1)`) mide qué tan seguro está
# el modelo. Encuentren las 5 imágenes con **menor** confianza y muéstrenlas con
# sus dos clases más probables. ¿Son casos realmente ambiguos? ¿Usarían un
# umbral de confianza para decidir cuándo pedir la opinión de una persona?

# %%
# [Su código aquí]

# %% [markdown]
# ---
# ## 🏠 Tarea: entrene su propia CNN de XRD en la GPU
#
# El modelo de hoy se entrenó solo con XRD **simulados** y luego lo probamos
# con difractogramas **reales**. ¿Qué parte del aumento de datos le ayuda a
# sobrevivir ese salto? La tarea es un experimento corto: entrenar la misma red
# 20 épocas (unos minutos en GPU) quitando una parte del aumento de datos, y
# comparar con los demás.
#
# 1. Abra `tareas/tarea_vision.sbatch` y elija `NOMBRE` y `AUMENTO`
#    (`completo`, `ninguno`, `sin_fondo`, `sin_kalfa2`, `sin_ruido` o
#    `sin_ensanchamiento`). Repartan las opciones entre el grupo.
# 2. Envíelo con la celda de abajo. El trabajo entra a la cola nukwa; no hay
#    que esperar frente a la pantalla.
# 3. Cuando termine, compare su exactitud en simulados y en reales con la del
#    modelo del curso. ¿Qué parte del aumento importó más para los reales?

# %%
# Enviar el entrenamiento a la cola nukwa (un solo trabajo de GPU a la vez por estudiante)
import sys
CURSO = DATOS.parent
sys.path.insert(0, str(CURSO / "tareas"))
from gpu import enviar_a_gpu, estado_gpu
enviar_a_gpu("tareas/tarea_vision.sbatch")

# %%
# PENDING = esperando GPU, RUNNING = corriendo. Si no aparece, ya terminó.
estado_gpu()

# %%
MI_ENTRENAMIENTO = "sin_fondo"      # el NOMBRE que puso en el .sbatch
carpeta = CURSO / "salidas_vision" / MI_ENTRENAMIENTO
if (carpeta / "info.json").exists():
    info_mia = json.load(open(carpeta / "info.json"))
    mio = CNN1D(len(SISTEMAS))
    mio.load_state_dict(torch.load(carpeta / "xrd_cnn1d.pt", map_location="cpu")); mio.eval()
    with torch.no_grad():                # mismos datos de prueba que usamos en clase
        acc_sim = accuracy_score(y_sim, mio(torch.tensor(X_sim)).argmax(1).numpy())
        acc_real = accuracy_score(y_real, mio(torch.tensor(X_real)).argmax(1).numpy())
    print(f"Su modelo ({info_mia['opciones']['aumento']}, {info_mia['opciones']['epocas']} épocas, {info_mia['segundos']} s):")
    print(f"  simulados: {acc_sim:.1%}   reales: {acc_real:.1%}")
    print("Modelo del curso (aumento completo, 150 épocas):")
    print(f"  simulados: {accuracy_score(y_sim, pred_sim):.1%}   reales: {accuracy_score(y_real, pred_real):.1%}")
    plt.plot(range(1, len(info_mia["historial"]) + 1), [h["val"] for h in info_mia["historial"]], marker="o")
    plt.xlabel("época"); plt.ylabel("exactitud de validación"); plt.title(f"Entrenamiento: {MI_ENTRENAMIENTO}")
    plt.show()
else:
    print("Todavía no hay resultados. Revise estado_gpu() y los archivos .out en tareas/registros/")

# %% [markdown]
# ---
# ## 🏠 Ejercicio (para casa)
#
# **Pruebe el modelo XRD con un difractograma suyo** (Cu Kα, archivo de dos
# columnas 2θ e intensidad, como los que exporta el difractómetro):
#
# 1. Léalo con `np.loadtxt`, interpólelo a `x2t` con `np.interp`, reste el
#    mínimo y normalice a máximo 1.
# 2. Prediga con `predecir_xrd` y muestre las tres clases más probables.
# 3. Compare con lo que usted sabe de la muestra. Si no coincide, ¿cuál de las
#    causas del cambio de dominio sospecha?
#
# Si no tiene un patrón propio, use cualquiera de `xrd_reales_opxrd.npz`
# que el modelo haya fallado y explique por qué cree que falló.
#
# ---
# ### 🔜 Conexión con el Día 4
# > Hoy los modelos ya venían entrenados. Mañana volvemos al laboratorio: cómo
# > decidir **qué experimento hacer** cuando cada medición cuesta.
