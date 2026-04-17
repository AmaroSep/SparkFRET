"""
Analisis de patrones en sparkles FRET.

Lee sparkle_features.csv (salida de characterize_sparkles.py) y produce:

  NIVEL 1 — Por sparkle:
    - Matriz de correlacion entre features
    - PCA coloreado por Disease / Fraction
    - UMAP coloreado por Disease / Fraction / Patient

  NIVEL 2 — Por paciente (agrega sparkles a mediana por Sample):
    - Heatmap jerarquico (pacientes x features)
    - PCA de pacientes
    - Random Forest: ranking de features discriminantes

  NIVEL 3 — Estadistica por grupo:
    - Mann-Whitney entre todos los pares de grupos
    - Tabla de p-values y effect size (rank-biserial r)

Uso: D:/Cellpose/venv/Scripts/python D:/Cellpose/analyze_patterns.py
"""

import sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.cluster import hierarchy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import umap
import tkinter as tk
from tkinter import filedialog, messagebox

warnings.filterwarnings("ignore")

# ── GUI ───────────────────────────────────────────────────────────────────────
root = tk.Tk()
root.withdraw()

messagebox.showinfo(
    "SparkFRET — Analisis de Patrones",
    "Selecciona:\n"
    "  1. sparkle_features.csv\n"
    "  2. Carpeta de salida"
)

feat_path = filedialog.askopenfilename(
    title="1/2 — sparkle_features.csv",
    initialdir="D:/Cellpose/results",
    filetypes=[("CSV", "*.csv"), ("Todos", "*.*")]
)
if not feat_path: print("Cancelado."); sys.exit(0)

out_dir = filedialog.askdirectory(
    title="2/2 — Carpeta de salida",
    initialdir=str(Path(feat_path).parent)
)
if not out_dir: print("Cancelado."); sys.exit(0)
root.destroy()

OUT = Path(out_dir)
OUT.mkdir(parents=True, exist_ok=True)

# ── Carga y prepara datos ─────────────────────────────────────────────────────
print("Cargando datos...")
df = pd.read_csv(feat_path)
print(f"  Sparkles totales: {len(df)}")
print(f"  Columnas: {list(df.columns)}")

# Features numericas de interes
CANDIDATE_FEATURES = [
    "area_px", "diameter_px", "circularity", "eccentricity",
    "solidity", "compactness", "major_axis_px", "minor_axis_px",
    "fret_mean", "fret_integrated", "fret_max", "fret_median",
    "fret_over_cell",
]
FEATURES = [f for f in CANDIDATE_FEATURES if f in df.columns]
print(f"\nFeatures disponibles: {FEATURES}")

# Columnas de grupo
META = {c: c for c in ["Disease", "Fraction", "Sample", "Replicate"]
        if c in df.columns}

DISEASE_ORDER  = ["Control", "AsymAD", "AD"]
DISEASE_COLORS = {"Control": "#4CAF50", "AsymAD": "#FF9800", "AD": "#F44336"}
FRAC_MARKERS   = {"F9": "o", "Sarkosyl": "s"}

# Limpia: solo filas con Disease conocida y todas las features
df_clean = df.dropna(subset=["Disease"] + FEATURES).copy()
diseases  = [d for d in DISEASE_ORDER if d in df_clean["Disease"].unique()]
fractions = sorted(df_clean["Fraction"].dropna().unique()) if "Fraction" in df_clean.columns else []

print(f"\nSparkles validos: {len(df_clean)}")
print(f"Groups Disease: {diseases}")
print(f"Fracciones: {fractions}")

X      = df_clean[FEATURES].values
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

# ─────────────────────────────────────────────────────────────────────────────
# NIVEL 1A — Correlacion entre features
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/6] Matriz de correlacion...")
corr = pd.DataFrame(X_sc, columns=FEATURES).corr()

fig, ax = plt.subplots(figsize=(len(FEATURES) * 0.9 + 1, len(FEATURES) * 0.9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, annot_kws={"size": 8})
ax.set_title("Correlacion entre features de sparkles", fontsize=13,
             fontweight="bold", pad=12)
plt.tight_layout()
fig.savefig(OUT / "correlation_matrix.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "correlation_matrix.svg",          bbox_inches="tight")
plt.close(fig)
print("  → correlation_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
# NIVEL 1B — PCA por sparkle
# ─────────────────────────────────────────────────────────────────────────────
print("[2/6] PCA por sparkle...")
pca  = PCA(n_components=min(len(FEATURES), 4))
X_pca = pca.fit_transform(X_sc)
var   = pca.explained_variance_ratio_ * 100

# Subsample para grafico si hay muchos puntos
MAX_PLOT = 5000
idx_plot = (np.random.choice(len(df_clean), MAX_PLOT, replace=False)
            if len(df_clean) > MAX_PLOT else np.arange(len(df_clean)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PCA — Sparkles individuales", fontsize=13, fontweight="bold")

for ax_i, (ax, frac_filter) in enumerate(zip(axes, (None, None))):
    color_by = "Disease" if ax_i == 0 else "Fraction"
    if color_by not in df_clean.columns:
        ax.set_visible(False)
        continue

    vals = df_clean[color_by].iloc[idx_plot].values
    uniq = sorted(df_clean[color_by].dropna().unique())
    palette = (DISEASE_COLORS if color_by == "Disease"
               else {v: plt.cm.tab10(i) for i, v in enumerate(uniq)})

    for v in (diseases if color_by == "Disease" else uniq):
        mask = vals == v
        ax.scatter(X_pca[idx_plot][mask, 0],
                   X_pca[idx_plot][mask, 1],
                   c=palette.get(v, "gray"), label=v,
                   alpha=0.4, s=8, edgecolors="none")

    ax.set_xlabel(f"PC1 ({var[0]:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var[1]:.1f}%)", fontsize=10)
    ax.set_title(f"Coloreado por {color_by}", fontsize=11)
    ax.legend(fontsize=9, markerscale=2, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(OUT / "pca_sparkles.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "pca_sparkles.svg",          bbox_inches="tight")
plt.close(fig)
print("  → pca_sparkles.png")

# Loadings de PCA
loadings = pd.DataFrame(pca.components_.T, index=FEATURES,
                         columns=[f"PC{i+1}" for i in range(pca.n_components_)])
loadings["PC1_abs"] = loadings["PC1"].abs()
print(f"\n  Varianza explicada: PC1={var[0]:.1f}% PC2={var[1]:.1f}%")
print("  Top features PC1:")
print(loadings.sort_values("PC1_abs", ascending=False)[["PC1", "PC2"]].head(5).round(3).to_string())
loadings.to_csv(OUT / "pca_loadings.csv")

# ─────────────────────────────────────────────────────────────────────────────
# NIVEL 1C — UMAP por sparkle
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] UMAP por sparkle (puede tardar 1-2 min)...")
N_UMAP = min(len(df_clean), 8000)
idx_umap = (np.random.choice(len(df_clean), N_UMAP, replace=False)
            if len(df_clean) > N_UMAP else np.arange(len(df_clean)))

reducer  = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                     random_state=42, verbose=False)
X_umap   = reducer.fit_transform(X_sc[idx_umap])
df_umap  = df_clean.iloc[idx_umap].copy()
df_umap["UMAP1"] = X_umap[:, 0]
df_umap["UMAP2"] = X_umap[:, 1]

ncols = 1 + len(fractions)
fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), squeeze=False)
fig.suptitle("UMAP — Sparkles individuales", fontsize=13, fontweight="bold")

# Panel 1: Disease
ax = axes[0, 0]
for d in diseases:
    sub = df_umap[df_umap["Disease"] == d]
    ax.scatter(sub["UMAP1"], sub["UMAP2"],
               c=DISEASE_COLORS.get(d, "gray"), label=d,
               alpha=0.4, s=8, edgecolors="none")
ax.set_title("Coloreado por Disease", fontsize=11)
ax.legend(fontsize=9, markerscale=2, frameon=False)
ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Paneles por fraccion
for ci, frac in enumerate(fractions, start=1):
    ax = axes[0, ci]
    sub_frac = df_umap[df_umap["Fraction"] == frac] if "Fraction" in df_umap.columns else df_umap
    for d in diseases:
        sub = sub_frac[sub_frac["Disease"] == d]
        ax.scatter(sub["UMAP1"], sub["UMAP2"],
                   c=DISEASE_COLORS.get(d, "gray"), label=d,
                   alpha=0.5, s=10, edgecolors="none")
    ax.set_title(f"Fraccion: {frac}", fontsize=11)
    ax.legend(fontsize=9, markerscale=2, frameon=False)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(OUT / "umap_sparkles.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "umap_sparkles.svg",          bbox_inches="tight")
plt.close(fig)
df_umap.to_csv(OUT / "umap_coordinates.csv", index=False)
print("  → umap_sparkles.png")

# ─────────────────────────────────────────────────────────────────────────────
# NIVEL 2 — Agrega por paciente (mediana de sparkles)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] Heatmap jerarquico por paciente...")

group_by = [c for c in ["Sample", "Disease", "Fraction"] if c in df_clean.columns]
patient  = (df_clean.groupby(group_by)[FEATURES]
            .median()
            .reset_index())
patient  = patient.dropna(subset=FEATURES)

print(f"  Pacientes con datos: {len(patient)}")

# Normaliza por feature (Z-score entre pacientes)
feat_matrix = patient[FEATURES].values
feat_z      = StandardScaler().fit_transform(feat_matrix)
feat_z_df   = pd.DataFrame(feat_z, columns=FEATURES)

# Etiqueta para las filas: Sample + Disease
if "Sample" in patient.columns and "Disease" in patient.columns:
    row_labels = [f"{row.Sample}\n({row.Disease})"
                  for _, row in patient.iterrows()]
elif "Sample" in patient.columns:
    row_labels = patient["Sample"].astype(str).tolist()
else:
    row_labels = [str(i) for i in range(len(patient))]

# Color de filas por disease
row_colors = pd.Series(
    [DISEASE_COLORS.get(d, "#AAAAAA")
     for d in patient.get("Disease", [""] * len(patient))],
    name="Disease"
)

fig = sns.clustermap(
    feat_z_df,
    row_colors=row_colors,
    col_cluster=True,
    row_cluster=True,
    cmap="RdBu_r",
    center=0,
    yticklabels=row_labels,
    xticklabels=FEATURES,
    figsize=(max(10, len(FEATURES) * 0.9), max(8, len(patient) * 0.4)),
    cbar_pos=(0.02, 0.8, 0.03, 0.15),
    dendrogram_ratio=0.15,
    annot=False,
)
fig.ax_heatmap.set_xticklabels(
    fig.ax_heatmap.get_xticklabels(), rotation=40, ha="right", fontsize=9)
fig.ax_heatmap.set_yticklabels(
    fig.ax_heatmap.get_yticklabels(), fontsize=7)
fig.fig.suptitle("Heatmap jerarquico — pacientes x features",
                 fontsize=13, fontweight="bold", y=1.02)

# Leyenda disease
patches = [mpatches.Patch(color=DISEASE_COLORS.get(d, "gray"), label=d)
           for d in diseases]
fig.fig.legend(handles=patches, loc="upper right", fontsize=9,
               bbox_to_anchor=(1.12, 1.0), frameon=False)

fig.savefig(OUT / "patient_heatmap.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "patient_heatmap.svg",          bbox_inches="tight")
plt.close("all")
patient.to_csv(OUT / "patient_median_features.csv", index=False)
print("  → patient_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# NIVEL 2B — Random Forest feature importance
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Random Forest — importancia de features...")

# Usa sparkles individuales; entrena Control vs AD (los extremos)
df_rf = df_clean[df_clean["Disease"].isin(["Control", "AD"])].copy()
if len(df_rf) > 500:
    df_rf = df_rf.sample(n=min(len(df_rf), 10000), random_state=42)

if len(df_rf) >= 20:
    X_rf = df_rf[FEATURES].values
    y_rf = (df_rf["Disease"] == "AD").astype(int).values

    rf = RandomForestClassifier(n_estimators=200, max_depth=6,
                                 random_state=42, n_jobs=-1)
    cv_scores = cross_val_score(rf, X_rf, y_rf, cv=5, scoring="roc_auc")
    rf.fit(X_rf, y_rf)

    importance = pd.DataFrame({
        "feature":    FEATURES,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)

    print(f"  AUC CV (Control vs AD): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print("  Feature importance:")
    print(importance.to_string(index=False))
    importance.to_csv(OUT / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, len(FEATURES) * 0.5 + 1))
    colors = ["#E53935" if i == 0 else "#1E88E5" if i == 1 else "#888888"
              for i in range(len(importance))]
    ax.barh(importance["feature"][::-1],
            importance["importance"][::-1],
            color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Importancia (Gini)", fontsize=11)
    ax.set_title(f"Random Forest — Control vs AD\n"
                 f"AUC = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "feature_importance.svg",          bbox_inches="tight")
    plt.close(fig)
    print("  → feature_importance.png")
else:
    print("  SKIP: pocos datos para Random Forest")

# ─────────────────────────────────────────────────────────────────────────────
# NIVEL 3 — Estadistica por grupo: p-values + effect size
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] Tests estadisticos entre grupos...")

def rank_biserial_r(x, y):
    """Effect size para Mann-Whitney (rango biserial r)."""
    nx, ny = len(x), len(y)
    u, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return 1 - (2 * u) / (nx * ny)

pairs = [(diseases[i], diseases[j])
         for i in range(len(diseases))
         for j in range(i + 1, len(diseases))]

rows = []
for feat in FEATURES:
    for g1, g2 in pairs:
        for frac in (fractions if fractions else [None]):
            subset = df_clean
            if frac is not None and "Fraction" in subset.columns:
                subset = subset[subset["Fraction"] == frac]
            v1 = subset[subset["Disease"] == g1][feat].dropna()
            v2 = subset[subset["Disease"] == g2][feat].dropna()
            if len(v1) < 3 or len(v2) < 3:
                continue
            _, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
            r    = rank_biserial_r(v1.values, v2.values)
            rows.append({
                "feature":   feat,
                "group1":    g1,
                "group2":    g2,
                "fraction":  frac or "all",
                "n1":        len(v1),
                "n2":        len(v2),
                "p_value":   round(p, 6),
                "effect_r":  round(r, 3),
                "significant": p < 0.05,
            })

stats_df = pd.DataFrame(rows)

# Correccion Benjamini-Hochberg por fraccion
from scipy.stats import false_discovery_control
if len(stats_df) > 0:
    try:
        stats_df["p_adj"] = false_discovery_control(stats_df["p_value"])
    except Exception:
        stats_df["p_adj"] = stats_df["p_value"] * len(stats_df)  # Bonferroni

stats_df.to_csv(OUT / "statistical_tests.csv", index=False)

# Imprime los significativos
sig = stats_df[stats_df["significant"]].sort_values("p_value")
print(f"\n  Features significativas (p<0.05): {len(sig)} de {len(stats_df)} comparaciones")
if len(sig) > 0:
    print(sig[["feature", "group1", "group2", "fraction",
               "p_value", "effect_r"]].to_string(index=False))

# Heatmap de -log10(p) por feature x comparacion
if len(stats_df) > 0:
    stats_df["comparison"] = (stats_df["group1"] + " vs " + stats_df["group2"]
                               + " [" + stats_df["fraction"] + "]")
    pivot = stats_df.pivot_table(
        index="feature", columns="comparison",
        values="p_value", aggfunc="min"
    )
    logp = -np.log10(pivot.clip(lower=1e-10))

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 2),
                                    max(5, len(FEATURES) * 0.5 + 1)))
    sns.heatmap(logp, annot=pivot.round(4), fmt=".4f",
                cmap="YlOrRd", ax=ax, linewidths=0.5,
                cbar_kws={"label": "-log10(p-value)"},
                annot_kws={"size": 8})
    ax.set_title("-log₁₀(p-value) por feature y comparacion\n"
                 "(valores altos = mas significativo)",
                 fontsize=12, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT / "pvalue_heatmap.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "pvalue_heatmap.svg",          bbox_inches="tight")
    plt.close(fig)
    print("  → pvalue_heatmap.png")

# ── Resumen final ─────────────────────────────────────────────────────────────
print(f"""
=== ARCHIVOS GENERADOS ===
  correlation_matrix.png     — relacion entre features
  pca_sparkles.png           — PCA individual por sparkle
  pca_loadings.csv           — contribucion de features a PCs
  umap_sparkles.png          — UMAP individual por sparkle
  umap_coordinates.csv       — coordenadas UMAP por sparkle
  patient_heatmap.png        — clustering jerarquico por paciente
  patient_median_features.csv— mediana de features por paciente
  feature_importance.png     — ranking Random Forest (Control vs AD)
  feature_importance.csv
  statistical_tests.csv      — p-values y effect size por comparacion
  pvalue_heatmap.png         — resumen visual de significancia

Carpeta: {OUT}
""")
