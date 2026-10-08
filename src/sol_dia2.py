# %% [markdown]
# # ✅ Solución — Ejercicio del Día 2
#
# *Notebook del instructor.* Predicción de bandgap desde features Magpie,
# usando el caché de Materials Project del Día 2.


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
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

def encontrar_datos() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents[:3]]:
        if (base / "datos").is_dir():
            return base / "datos"
    raise FileNotFoundError("No encuentro la carpeta datos/")

DATOS = encontrar_datos()
df_mp = pd.read_json(DATOS / "mp_bandgap_cache.json")
print(f"Caché MP: {df_mp.shape}")

# %% [markdown]
# ## 1. Materiales con bandgap 1.0–2.0 eV + scatter vs energía de formación

# %%
sel = df_mp[(df_mp["band_gap"] >= 1.0) & (df_mp["band_gap"] <= 2.0)].copy()
print(f"Materiales con bandgap 1.0-2.0 eV: {len(sel)}")

plt.figure(figsize=(7, 5))
plt.scatter(sel["band_gap"], sel["Ef"], alpha=0.4, s=15)
plt.xlabel("Bandgap (eV)")
plt.ylabel("Energía de formación (eV/átomo)")
plt.title(f"Candidatos con gap 1.0–2.0 eV (n={len(sel)})")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. RandomForest para predecir bandgap desde Magpie

# %%
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition

stc = StrToComposition(target_col_id="composition")
import joblib
stc.set_n_jobs(joblib.cpu_count())  # solo los núcleos de esta sesión, no los 40 del nodo
sel = stc.featurize_dataframe(sel, "formula", ignore_errors=True, pbar=False)
sel = sel.dropna(subset=["composition"])

ep = ElementProperty.from_preset("magpie")
ep.set_n_jobs(joblib.cpu_count())  # solo los núcleos de esta sesión, no los 40 del nodo
sel = ep.featurize_dataframe(sel, col_id="composition", ignore_errors=True, pbar=False)

feature_cols = [c for c in sel.columns if "MagpieData" in c]
X = sel[feature_cols].dropna()
y = sel.loc[X.index, "band_gap"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
rf.fit(X_train_sc, y_train)
y_pred = rf.predict(X_test_sc)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.3f} eV  |  R²: {r2:.3f}")
# Con gaps experimentales/DFT en rango estrecho (1-2 eV), esperar MAE ~0.2-0.4 eV
# y R² modesto: el rango del target es solo 1 eV, no hay mucha varianza que explicar.

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, y_pred, alpha=0.5, s=20)
lims = [y.min(), y.max()]
ax.plot(lims, lims, "--", color=GRIS, lw=1)
ax.set_xlabel("Bandgap real (eV)")
ax.set_ylabel("Bandgap predicho (eV)")
ax.set_title(f"Parity — MAE={mae:.2f} eV, R²={r2:.2f}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Feature más importante y su sentido físico

# %%
importances = pd.Series(rf.feature_importances_, index=feature_cols)
print(importances.nlargest(10))

importances.nlargest(10).sort_values().plot(kind="barh", figsize=(8, 5), edgecolor="white")
plt.title("Top 10 features para predecir bandgap")
plt.tight_layout()
plt.show()

# Comentario esperado: dominan la electronegatividad (media/diferencia) y los
# electrones de valencia (NValence, NsValence...). Tiene sentido físico: el gap
# nace del carácter iónico/covalente del enlace — a mayor diferencia de
# electronegatividad, más iónico el enlace y mayor el gap; el llenado de la
# banda de valencia depende directamente del conteo de electrones de valencia.

# %% [markdown]
# ---
# # Retos en clase — Día 2
#
# ## Reto 1: validación cruzada + bajar el MAE ajustando hiperparámetros
#
# Requiere el estado del notebook del día (X_train_sc, y_train...). Aquí lo
# reconstruimos de forma autocontenida sobre el pickle del D1.

# %%
df1 = pd.read_pickle(DATOS / "df_con_features_dia1.pkl")
fc = [c for c in df1.columns if "MagpieData" in c]
Xd = df1[fc].dropna()
yd = df1.loc[Xd.index, "K_VRH"]
X_tr, X_te, y_tr, y_te = train_test_split(Xd, yd, test_size=0.2, random_state=42)
sc = StandardScaler()
X_tr_sc, X_te_sc = sc.fit_transform(X_tr), sc.transform(X_te)

# Parte 1 — cross_val_score
from sklearn.model_selection import cross_val_score

rf_base = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
scores = cross_val_score(rf_base, X_tr_sc, y_tr, cv=5, scoring="r2", n_jobs=-1)
print(f"R² por fold: {np.round(scores, 3)} → {scores.mean():.3f} ± {scores.std():.3f}")

# Extra: folds que no separan polimorfos (misma fórmula reducida en un solo lado)
from sklearn.model_selection import GroupKFold
from pymatgen.core import Composition
grupos = df1.loc[X_tr.index, "formula"].apply(lambda f: Composition(f).reduced_formula)
scores_g = cross_val_score(rf_base, X_tr_sc, y_tr, groups=grupos,
                           cv=GroupKFold(n_splits=5), scoring="r2", n_jobs=-1)
print(f"R² con GroupKFold: {scores_g.mean():.3f} ± {scores_g.std():.3f}")
# Si baja respecto al anterior, parte del R² venía de "reconocer" polimorfos.

# %%
# Parte 2 — hiperparámetros. Lo que suele descubrir la sala:
configs = {
    "base (200 árboles)":        dict(n_estimators=200),
    "500 árboles":               dict(n_estimators=500),
    "500 + max_features=0.3":    dict(n_estimators=500, max_features=0.3),
    "500 + max_features='sqrt'": dict(n_estimators=500, max_features="sqrt"),
}
for nombre, kw in configs.items():
    m = RandomForestRegressor(n_jobs=-1, random_state=42, **kw).fit(X_tr_sc, y_tr)
    mae_c = mean_absolute_error(y_te, m.predict(X_te_sc))
    print(f"{nombre:28s} MAE = {mae_c:.1f} GPa")
# Meta del reto: MAE < 14 GPa (con max_features=0.3 se llega a ~13.9).
# Lección: más árboles ayuda poco pasado ~300; max_features es la palanca real
# en datasets con features correlacionadas. La mejora total es ~1-2 GPa: los
# hiperparámetros afinan, no transforman. (Ganancias grandes vienen de mejores
# features o más datos.)

# %% [markdown]
# ## Reto 2: pool ternario A-B-C

# %%
from pymatgen.core import Composition as Comp

el_A = ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W"]
el_B = ["C", "N", "B"]
el_C = ["Al", "Si"]

tern = []
for A in el_A:
    for B in el_B:
        for C in el_C:
            for f in [f"{A}2{B}1{C}1", f"{A}1{B}1{C}1", f"{A}3{B}2{C}1"]:
                c = Comp(f)
                tern.append({"formula": f, "reduced": c.reduced_formula, "composition": c})
df_t = pd.DataFrame(tern).drop_duplicates("reduced").reset_index(drop=True)

ep2 = ElementProperty.from_preset("magpie")
ep2.set_n_jobs(joblib.cpu_count())  # solo los núcleos de esta sesión, no los 40 del nodo
df_t = ep2.featurize_dataframe(df_t, col_id="composition", ignore_errors=True, pbar=False)
fc_t = [c for c in df_t.columns if "MagpieData" in c]

rf_full = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42).fit(X_tr_sc, y_tr)
Xt = df_t[fc_t].dropna()
df_t.loc[Xt.index, "K_pred"] = rf_full.predict(sc.transform(Xt))
print(df_t.nlargest(10, "K_pred")[["reduced", "K_pred"]].to_string(index=False))

# Discusión esperada: sí aparecen ternarios >300 GPa (típicamente ricos en W/C).
# CONFIANZA: menor que en binarios — el training set (elastic_tensor_2015)
# contiene pocos ternarios de esta familia; el modelo extrapola químicamente.
# La predicción sirve para PRIORIZAR, no para prometer.
