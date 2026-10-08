# %% [markdown]
# # ✅ Solución de referencia — Brief A: Termoeléctricos
#
# *Notebook del instructor. Una de MUCHAS soluciones válidas — sirve como vara
# de comparación, no como respuesta única.*
#
# **Estrategia elegida:** frente de Pareto ZT-costo + modelo RF para entender
# qué mueve el ZT + lista corta balanceada por familias químicas.


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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def encontrar_datos() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents[:3]]:
        if (base / "datos").is_dir():
            return base / "datos"
    raise FileNotFoundError("No encuentro la carpeta datos/")

DATOS = encontrar_datos()
df = pd.read_csv(DATOS / "brief_A_termoelectricos.csv")
print(df.shape)

# %% [markdown]
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>Sección 1 — EDA</b> </div>

# %%
print(df["grupo"].value_counts())
df[["ZT_estimated", "costo_relativo", "bandgap_eV", "kappa_W_mK"]].describe().round(2)

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
df["ZT_estimated"].hist(bins=40, ax=axes[0], edgecolor="white")
axes[0].set(title="Distribución de ZT estimado", xlabel="ZT")
orden_zt = df.groupby("grupo")["ZT_estimated"].median().sort_values()
df.boxplot(column="ZT_estimated", by="grupo", ax=axes[1], rot=60)
axes[1].set_title("ZT por familia química")
plt.suptitle("")
plt.tight_layout()
plt.show()

# %% [markdown]
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>Sección 2 — Modelo: ¿qué físico-química mueve el ZT?</b> </div>
#
# No necesitamos el modelo para *rankear* (ZT ya viene en la tabla) — lo
# usamos para **entender** y para validar consistencia.
#
# **El momento clave: feature engineering con física.** La figura de mérito
# es ZT = S²σT/κ. Si el equipo construye el feature S²σ/κ (factor de potencia
# sobre conductividad térmica), el modelo debería mejorar drásticamente —
# la física correcta vale más que 100 features genéricas.

# %%
# Modelo 1: columnas "crudas"
feats_base = ["bandgap_eV", "seebeck_uV_K", "sigma_S_m", "kappa_W_mK"] + \
             [c for c in df.columns if "Magpie" in c or c.startswith("magpie")]
feats_base = [c for c in feats_base if c in df.columns]

# Modelo 2: + feature de física (S en µV/K → V/K)
df["PF_sobre_kappa"] = (df["seebeck_uV_K"] * 1e-6) ** 2 * df["sigma_S_m"] / df["kappa_W_mK"]
feats_fisica = feats_base + ["PF_sobre_kappa"]

y = df["ZT_estimated"]
for nombre, feats in [("crudo", feats_base), ("+ S²σ/κ", feats_fisica)]:
    X_tr, X_te, y_tr, y_te = train_test_split(df[feats], y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    pred = rf.predict(X_te)
    print(f"Modelo {nombre:10s} MAE: {mean_absolute_error(y_te, pred):.3f}  R²: {r2_score(y_te, pred):.3f}")

pd.Series(rf.feature_importances_, index=feats_fisica).nlargest(8).sort_values().plot(
    kind="barh", figsize=(7, 4), edgecolor="white",
    title="¿Qué mueve el ZT? (modelo con feature de física)")
plt.tight_layout()
plt.show()
# El feature S²σ/κ debe dominar la importancia — el modelo "confirma" la
# física. Los equipos que solo usen features genéricas verán R² menor:
# ese contraste es una de las lecciones del brief.

# %% [markdown]
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>Sección 3 — Resultado: frente de Pareto ZT vs costo y lista corta</b> </div>

# %%
def es_pareto(zt, costo):
    """True si ningún otro material tiene mejor ZT y menor costo a la vez."""
    flags = np.ones(len(zt), dtype=bool)
    for i in range(len(zt)):
        flags[i] = not ((zt > zt[i]) & (costo < costo[i])).any()
    return flags

df["pareto"] = es_pareto(df["ZT_estimated"].values, df["costo_relativo"].values)
pareto = df[df["pareto"]].sort_values("ZT_estimated", ascending=False)
print(f"Materiales en el frente de Pareto: {len(pareto)}")

# Lista corta: los Pareto de mayor ZT, máximo 3 por familia (diversificar
# riesgo de síntesis) — criterio defendible, no único
top10 = (pareto.groupby("grupo").head(3)
         .nlargest(10, "ZT_estimated")
         [["formula", "grupo", "ZT_estimated", "costo_relativo", "bandgap_eV", "kappa_W_mK"]])
print(top10.to_string(index=False))

# %% [markdown]
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>Sección 4 — Comunicación</b> </div>

# %%
plt.figure(figsize=(8, 6))
plt.scatter(df["costo_relativo"], df["ZT_estimated"], alpha=0.25, s=15, label="Catálogo (800)")
plt.scatter(pareto["costo_relativo"], pareto["ZT_estimated"], c="tab:orange", s=40,
            label=f"Frente de Pareto ({len(pareto)})")
sel = df["formula"].isin(top10["formula"])
plt.scatter(df.loc[sel, "costo_relativo"], df.loc[sel, "ZT_estimated"],
            c=NARANJA, marker="*", s=200, label="Lista corta (10)")
plt.xlabel("Costo relativo")
plt.ylabel("ZT estimado")
plt.title("Selección de candidatos termoeléctricos: desempeño vs costo")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Párrafo para el cliente (ejemplo):**
#
# > De los 800 materiales del catálogo, 10 ofrecen la mejor combinación de
# > eficiencia termoeléctrica proyectada y costo. Diversificamos entre 4
# > familias químicas para reducir el riesgo de que una sola ruta de síntesis
# > falle. Los tres primeros justifican validación experimental inmediata; el
# > resto son respaldo si la primera tanda decepciona.
#
# <div style="background:#1A2E4A;color:white;padding:10px 18px;border-radius:8px;margin-top:14px;"><b>Sección 5 — Limitaciones (lo que el equipo DEBE decir)</b> </div>
#
# - `ZT_estimated` es un **estimado de modelo**: la literatura muestra que
#   estos estimados sobreestiman el ZT medido ~2× (visto en teoría, T5.2).
#   La lista corta ordena bien, pero los valores absolutos no son promesas.
# - El costo relativo no incluye procesabilidad ni toxicidad (varios
#   candidatos contienen Pb/Te — bandera regulatoria para automotriz).
# - Validación siguiente: sintetizar 2-3, medir ZT real a 400-700 °C, y
#   **re-entrenar con esos datos** (active learning, Día 4).
