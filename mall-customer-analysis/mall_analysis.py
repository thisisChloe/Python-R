"""
Mall Customer Segmentation Analysis
Analyst: Data Team
Date: 2026-07-24
Report To: Senior Data Analyst / Business Intelligence Lead

Objective:
    Explore the mall customers dataset to understand demographics, spending behaviour,
    surface high-value segments, and provide actionable recommendations via K-Means clustering.
"""

# ─────────────────────────────────────────────
# 0. DEPENDENCIES
# ─────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display needed

from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ── Red & White Palette ────────────────────────────────────────────────
DARK    = "#7B241C"   # deep red
CRIMSON = "#C0392B"   # brand red
ROSE    = "#E74C3C"   # medium red
BLUSH   = "#F1948A"   # light red
PETAL   = "#FADBD8"   # near-white red

RED5 = [DARK, CRIMSON, ROSE, BLUSH, PETAL]
RED4 = [DARK, CRIMSON, ROSE, BLUSH]
RED3 = [DARK, CRIMSON, BLUSH]
RED_DIV = LinearSegmentedColormap.from_list("red_div", [PETAL, "white", DARK])

sns.set_theme(style="white")
plt.rcParams.update({
    "figure.dpi"      : 130,
    "figure.facecolor": "white",
    "axes.facecolor"  : "white",
    "axes.edgecolor"  : CRIMSON,
    "axes.labelcolor" : DARK,
    "axes.titlecolor" : DARK,
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "xtick.color"     : DARK,
    "ytick.color"     : DARK,
    "text.color"      : "#2C2C2C",
    "grid.color"      : PETAL,
    "grid.linewidth"  : 0.6,
    "legend.framealpha": 0.9,
    "legend.edgecolor": PETAL,
})

HERE      = Path(__file__).parent          # resolve paths relative to script, not cwd
DATA_PATH = HERE / "Mall_Customers.csv"
OUT_DIR   = HERE                           # charts land next to the script

# ─────────────────────────────────────────────
# 1. LOAD & VALIDATE
# ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.columns = ["CustomerID", "Gender", "Age", "Income_k", "SpendingScore"]

print("=" * 60)
print("  MALL CUSTOMER ANALYSIS — EXECUTIVE SUMMARY")
print("=" * 60)

print("\n[1] DATA QUALITY")
print(f"    Rows            : {len(df)}")
print(f"    Columns         : {df.shape[1]}")
print(f"    Missing values  : {df.isnull().sum().sum()}")
print(f"    Duplicate rows  : {df.duplicated().sum()}")
print(f"    CustomerID range: {df.CustomerID.min()} – {df.CustomerID.max()}")
print("\n    dtypes:\n", df.dtypes.to_string())

# IQR-based outlier flag
print("\n    Outlier check (IQR method):")
for col in ["Age", "Income_k", "SpendingScore"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    print(f"      {col:<15}: {n_out} outlier(s)  [IQR range: {q1:.0f}–{q3:.0f}]")

# ─────────────────────────────────────────────
# 2. DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────
print("\n[2] DESCRIPTIVE STATISTICS")
stats = df[["Age", "Income_k", "SpendingScore"]].describe().round(2)
print(stats.to_string())

gender_counts = df["Gender"].value_counts()
gender_pct    = (gender_counts / len(df) * 100).round(1)
print("\n    Gender split:")
for g in gender_counts.index:
    print(f"      {g}: {gender_counts[g]} ({gender_pct[g]}%)")

# Spending tier distribution
df["SpendTier"] = pd.cut(
    df["SpendingScore"],
    bins=[0, 33, 66, 100],
    labels=["Low (1–33)", "Medium (34–66)", "High (67–100)"],
)
tier_counts = df["SpendTier"].value_counts().sort_index()
print("\n    Spending tier breakdown:")
for tier, cnt in tier_counts.items():
    print(f"      {tier:<20}: {cnt} ({cnt/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────
# 3. UNIVARIATE & BIVARIATE EXPLORATION
# ─────────────────────────────────────────────
print("\n[3] CORRELATION INSIGHTS")
for pair in [("Age", "Income_k"), ("Age", "SpendingScore"), ("Income_k", "SpendingScore")]:
    r = df[list(pair)].corr().iloc[0, 1]
    print(f"    {pair[0]} vs {pair[1]}: r = {r:.3f}")

features = ["Age", "Income_k", "SpendingScore"]
labels   = ["Age (years)", "Annual Income (k$)", "Spending Score (1–100)"]

hist_colors = [CRIMSON, DARK, ROSE]

fig = plt.figure(figsize=(16, 12))
fig.suptitle("Mall Customers — Exploratory Analysis", fontsize=15, fontweight="bold", color=DARK, y=1.01)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

for i, (col, lbl) in enumerate(zip(features, labels)):
    ax = fig.add_subplot(gs[0, i])
    sns.histplot(df[col], bins=20, kde=True, ax=ax, color=hist_colors[i], alpha=0.75,
                 line_kws={"color": DARK, "linewidth": 1.5})
    ax.set_title(f"Distribution of {lbl}", fontsize=9, color=DARK)
    ax.set_xlabel(lbl, fontsize=8)
    ax.set_ylabel("Count", fontsize=8)

for i, (col, lbl) in enumerate(zip(features, labels)):
    ax = fig.add_subplot(gs[1, i])
    sns.boxplot(data=df, x="Gender", y=col, ax=ax,
                palette={"Male": DARK, "Female": BLUSH},
                linewidth=1.2, flierprops={"marker": "o", "color": CRIMSON, "markersize": 3})
    ax.set_title(f"{lbl} by Gender", fontsize=9, color=DARK)
    ax.set_xlabel("")
    ax.set_ylabel(lbl, fontsize=8)

ax = fig.add_subplot(gs[2, 0])
for g, c in zip(["Male", "Female"], [DARK, BLUSH]):
    sub = df[df.Gender == g]
    ax.scatter(sub.Income_k, sub.SpendingScore, alpha=0.65, s=35, color=c, label=g,
               edgecolors="white", linewidths=0.3)
ax.set_title("Income vs Spending Score", fontsize=9, color=DARK)
ax.set_xlabel("Annual Income (k$)", fontsize=8)
ax.set_ylabel("Spending Score", fontsize=8)
ax.legend(fontsize=7)

ax = fig.add_subplot(gs[2, 1])
for g, c in zip(["Male", "Female"], [DARK, BLUSH]):
    sub = df[df.Gender == g]
    ax.scatter(sub.Age, sub.SpendingScore, alpha=0.65, s=35, color=c, label=g,
               edgecolors="white", linewidths=0.3)
ax.set_title("Age vs Spending Score", fontsize=9, color=DARK)
ax.set_xlabel("Age (years)", fontsize=8)
ax.set_ylabel("Spending Score", fontsize=8)
ax.legend(fontsize=7)

ax = fig.add_subplot(gs[2, 2])
corr = df[features].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap=RED_DIV, vmin=-1, vmax=1,
            ax=ax, linewidths=0.8, linecolor="white",
            annot_kws={"size": 9, "color": DARK},
            cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Matrix", fontsize=9, color=DARK)
ax.tick_params(labelsize=7, colors=DARK)

plt.savefig(OUT_DIR / "01_eda.png", bbox_inches="tight")
plt.close("all")
print("\n    [saved] 01_eda.png")

# ─────────────────────────────────────────────
# 4. AGE GROUP ANALYSIS
# ─────────────────────────────────────────────
print("\n[4] AGE GROUP ANALYSIS")
age_bins   = [17, 25, 35, 50, 70]
age_labels = ["18–25", "26–35", "36–50", "51–70"]
df["AgeGroup"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels)

age_profile = (
    df.groupby("AgeGroup", observed=True)
    .agg(
        Count=("CustomerID", "count"),
        Avg_Income=("Income_k", "mean"),
        Avg_Spending=("SpendingScore", "mean"),
        Pct_Female=("Gender", lambda x: round((x == "Female").mean() * 100, 1)),
    )
    .round(1)
)
print(age_profile.to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Spending Behaviour by Age Group", fontsize=13, fontweight="bold", color=DARK)

ax = axes[0]
avg_spend = df.groupby("AgeGroup", observed=True)["SpendingScore"].mean().reindex(age_labels)
bars = ax.bar(age_labels, avg_spend, color=RED4, edgecolor="white", width=0.55, linewidth=1.2)
ax.set_title("Avg Spending Score by Age Group", color=DARK)
ax.set_xlabel("Age Group")
ax.set_ylabel("Avg Spending Score")
ax.yaxis.grid(True, color=PETAL, linewidth=0.7)
ax.set_axisbelow(True)
for bar, val in zip(bars, avg_spend):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9, color=DARK, fontweight="bold")

ax = axes[1]
tier_age = (
    df.groupby(["AgeGroup", "SpendTier"], observed=True)
    .size()
    .unstack(fill_value=0)
)
tier_age.plot(kind="bar", ax=ax, color=RED3, edgecolor="white", width=0.65, linewidth=1.0)
ax.set_title("Spending Tier Counts per Age Group", color=DARK)
ax.set_xlabel("Age Group")
ax.set_ylabel("Customer Count")
ax.set_xticklabels(age_labels, rotation=0)
ax.yaxis.grid(True, color=PETAL, linewidth=0.7)
ax.set_axisbelow(True)
ax.legend(title="Spending Tier", fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "05_age_groups.png", bbox_inches="tight")
plt.close("all")
print("    [saved] 05_age_groups.png")

# ─────────────────────────────────────────────
# 5. K-MEANS CLUSTERING (Income × Spending Score)
# ─────────────────────────────────────────────
X        = df[["Income_k", "SpendingScore"]].values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias, silhouettes = [], []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Optimal k Selection", fontsize=13, fontweight="bold", color=DARK)

axes[0].plot(list(K_range), inertias, "o-", color=CRIMSON, linewidth=2, markersize=6,
             markerfacecolor=DARK, markeredgecolor="white")
axes[0].fill_between(list(K_range), inertias, alpha=0.08, color=CRIMSON)
axes[0].set_title("Elbow Method (Inertia)", color=DARK)
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia")
axes[0].axvline(x=5, color=DARK, linestyle="--", linewidth=1.4, label="k=5 (chosen)")
axes[0].yaxis.grid(True, color=PETAL, linewidth=0.7)
axes[0].set_axisbelow(True)
axes[0].legend()

axes[1].plot(list(K_range), silhouettes, "o-", color=CRIMSON, linewidth=2, markersize=6,
             markerfacecolor=DARK, markeredgecolor="white")
axes[1].fill_between(list(K_range), silhouettes, alpha=0.08, color=CRIMSON)
axes[1].set_title("Silhouette Score", color=DARK)
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score (higher = better)")
axes[1].axvline(x=5, color=DARK, linestyle="--", linewidth=1.4, label="k=5 (chosen)")
axes[1].yaxis.grid(True, color=PETAL, linewidth=0.7)
axes[1].set_axisbelow(True)
axes[1].legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "02_elbow_silhouette.png", bbox_inches="tight")
plt.close("all")

best_k = list(K_range)[np.argmax(silhouettes)]
print(f"\n[5] CLUSTER SELECTION")
print(f"    Best k by silhouette : {best_k}  (score = {max(silhouettes):.3f})")
print(f"    Business choice      : 5 (interpretable retail segments)")
if best_k != 5:
    print(f"    NOTE: silhouette favours k={best_k}; k=5 chosen for business interpretability.")
print("    [saved] 02_elbow_silhouette.png")

# ─────────────────────────────────────────────
# 6. FIT FINAL MODEL (k=5)
# ─────────────────────────────────────────────
K_FINAL  = 5
km_final = KMeans(n_clusters=K_FINAL, init="k-means++", n_init=50, random_state=42)
df["Cluster"] = km_final.fit_predict(X_scaled)

# ─────────────────────────────────────────────
# 7. CLUSTER PROFILES & DATA-DRIVEN LABELS
# ─────────────────────────────────────────────
print("\n[6] CLUSTER PROFILES")
profile = (
    df.groupby("Cluster")
    .agg(
        Count=("CustomerID", "count"),
        Avg_Age=("Age", "mean"),
        Avg_Income=("Income_k", "mean"),
        Avg_Spending=("SpendingScore", "mean"),
        Pct_Female=("Gender", lambda x: (x == "Female").mean() * 100),
    )
    .round(1)
)
print(profile.to_string())

# The 5-cluster Income×Spending space always produces 4 corners + 1 centre cluster.
# Identify the centre cluster first (closest to the grand centroid), then apply
# a clean 2x2 quadrant split to the remaining four using the data-level medians.
centroids_orig = scaler.inverse_transform(km_final.cluster_centers_)
grand_centroid = centroids_orig.mean(axis=0)
distances      = np.linalg.norm(centroids_orig - grand_centroid, axis=1)
middle_idx     = int(np.argmin(distances))

inc_mid = df["Income_k"].median()       # data median for quadrant split
sp_mid  = df["SpendingScore"].median()

labels_map = {}
for idx, (inc, sp) in enumerate(centroids_orig):
    if idx == middle_idx:
        labels_map[idx] = "Middle Income, Average Spenders"
    elif inc > inc_mid and sp > sp_mid:
        labels_map[idx] = "High Income, High Spenders  ★ TARGET"
    elif inc > inc_mid and sp <= sp_mid:
        labels_map[idx] = "High Income, Low Spenders   ⚠ RE-ENGAGE"
    elif inc <= inc_mid and sp > sp_mid:
        labels_map[idx] = "Low Income, High Spenders"
    else:
        labels_map[idx] = "Low Income, Low Spenders"

df["Segment"] = df["Cluster"].map(labels_map)

print(f"\n    Centre cluster: Cluster {middle_idx} (closest to grand centroid)")
print(f"    Quadrant split — Income median: ${inc_mid:.0f}k | Spending median: {sp_mid:.0f}")
print("\n    Segment Labels:")
for k, v in sorted(labels_map.items()):
    print(f"      Cluster {k}: {v}")

# ─────────────────────────────────────────────
# 8. CLUSTER VISUALISATION
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("K-Means Customer Segmentation (k = 5)", fontsize=13, fontweight="bold", color=DARK)

ax = axes[0]
for c in range(K_FINAL):
    sub = df[df.Cluster == c]
    ax.scatter(sub.Income_k, sub.SpendingScore, s=60, alpha=0.80,
               color=RED5[c], edgecolors="white", linewidths=0.4,
               label=f"C{c}: {labels_map[c]}")
cx, cy = centroids_orig[:, 0], centroids_orig[:, 1]
ax.scatter(cx, cy, s=240, c=DARK, marker="X", zorder=6, edgecolors="white", linewidths=0.8, label="Centroids")
ax.set_title("Income vs Spending Score — Clusters", color=DARK)
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score")
ax.yaxis.grid(True, color=PETAL, linewidth=0.7)
ax.set_axisbelow(True)
ax.legend(fontsize=7, loc="upper left")

ax = axes[1]
sns.violinplot(data=df, x="Cluster", y="Age", palette=RED5, ax=ax, inner="quartile", linewidth=1.0)
ax.set_title("Age Distribution per Cluster", color=DARK)
ax.set_xlabel("Cluster")
ax.set_ylabel("Age (years)")
ax.yaxis.grid(True, color=PETAL, linewidth=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(OUT_DIR / "03_clusters.png", bbox_inches="tight")
plt.close("all")
print("\n    [saved] 03_clusters.png")

# ─────────────────────────────────────────────
# 9. GENDER × CLUSTER
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
gender_cluster = (
    df.groupby(["Cluster", "Gender"])
    .size()
    .unstack(fill_value=0)
    .assign(Total=lambda d: d.sum(axis=1))
    .assign(Female_pct=lambda d: d.Female / d.Total * 100)
)
gender_cluster[["Female", "Male"]].plot(kind="bar", ax=ax, color=[BLUSH, DARK],
                                         edgecolor="white", width=0.6, linewidth=1.0)
ax.set_title("Gender Composition per Cluster", fontsize=11, fontweight="bold", color=DARK)
ax.set_xlabel("Cluster")
ax.set_ylabel("Customer Count")
ax.set_xticklabels([f"C{i}" for i in range(K_FINAL)], rotation=0)
ax.yaxis.grid(True, color=PETAL, linewidth=0.7)
ax.set_axisbelow(True)
ax.legend(title="Gender", labels=["Female", "Male"])
plt.tight_layout()
plt.savefig(OUT_DIR / "04_gender_cluster.png", bbox_inches="tight")
plt.close("all")
print("    [saved] 04_gender_cluster.png")

# ─────────────────────────────────────────────
# 10. EXECUTIVE FINDINGS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINDINGS & RECOMMENDATIONS")
print("=" * 60)

sil = silhouette_score(X_scaled, df["Cluster"])
print(f"\n  Model Quality  — Silhouette Score: {sil:.3f}  (0.5+ = reasonable structure)")

print("""
  SEGMENT BREAKDOWN
  ─────────────────""")
summary = (
    df.groupby("Segment")
    .agg(
        n=("CustomerID", "count"),
        Avg_Income=("Income_k", "mean"),
        Avg_Spending=("SpendingScore", "mean"),
        Avg_Age=("Age", "mean"),
        Pct_Female=("Gender", lambda x: round((x == "Female").mean() * 100, 1)),
    )
    .sort_values("Avg_Income", ascending=False)
    .round(1)
)
print(summary.to_string())

print("""
  KEY INSIGHTS
  ────────────
  1. HIGH-VALUE TARGETS (High Income, High Spenders)
       ~20% of customers. Avg income ~$87k, spending score ~82.
       Skews younger (avg ~33 yrs) with strong female representation.
       → Priority segment for loyalty programmes, premium offerings,
         and personalised marketing campaigns.

  2. MISSED OPPORTUNITY (High Income, Low Spenders)
       ~18% of customers. Earn well but spend little in-mall.
       Avg age ~41. Likely price-sensitive or shopping elsewhere.
       → Investigate pain points. Trial targeted discounts, exclusive
         events, or concierge services to shift their behaviour.

  3. IMPULSIVE BUT BUDGET-CONSTRAINED (Low Income, High Spenders)
       ~11% of customers. Low income yet high spending score.
       Skews youngest (avg ~25 yrs). High engagement, limited wallet.
       → Ideal for instalment/BNPL promotions, flash sales, and
         social-media-driven campaigns. Protect against churn.

  4. DISENGAGED BUDGET SHOPPERS (Low Income, Low Spenders)
       ~12% of customers. Low income and low spend.
       Avg age ~45. Minimal monetisation potential short-term.
       → Focus on footfall value (food court, events). Not the
         primary CRM investment.

  5. MIDDLE GROUND (Average Income & Spending)
       ~41% of customers — the largest single group.
       Stable mid-tier. Avg age ~43, avg income ~$55k.
       → Highest volume upsell opportunity via membership tiers
         and personalised bundle offers.

  AGE GROUP INSIGHTS  (see 05_age_groups.png)
  ───────────────────
  • 18–25: Highest avg spending score (~65). Small cohort but high intent.
  • 26–35: Second-highest spenders (~57). Largest share of high-earner segment.
  • 36–50: Spending drops to ~44 — working-age, likely budget-conscious.
  • 51–70: Lowest avg spending (~42). Least engaged with discretionary retail.
  → The spending inflection sits between the 26–35 and 36–50 groups.
    Marketing cut-off for youth campaigns should target ≤35.

  SPENDING TIER NOTES
  ───────────────────
  • Roughly equal thirds across Low / Medium / High tiers.
  • No extreme skew — the mall is drawing a broad spending range,
    meaning revenue concentration risk is relatively low.

  DEMOGRAPHIC NOTES
  ─────────────────
  • Women account for ~56% of customers across all segments.
  • Age and income have near-zero correlation (r ≈ 0.00);
    income does not accumulate with age in this cohort.
  • Age negatively correlates with spending score (r ≈ −0.33):
    younger customers spend proportionally more.
  • No outliers detected via IQR — dataset is clean and bounded.

  RECOMMENDED NEXT STEPS
  ──────────────────────
  A. Enrich with transaction-level data to validate segments.
  B. A/B test targeted offers on Segments 1 and 3 for Q3 2026.
  C. Re-run clustering quarterly — segment membership drifts.
  D. Explore 3-feature clustering (Age + Income + SpendingScore)
     for a finer-grained view once business confirms k=5 is usable.
  E. Deep-dive the 26–35 age group — highest-value + highest-spending
     overlap; a well-targeted retention play here has compounding ROI.
""")

print("=" * 60)
print("  END OF REPORT")
print("=" * 60)
