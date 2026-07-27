"""
Mall customer segmentation - basic EDA + KMeans clustering.

Loads the Mall_Customers dataset, looks at some quick stats, then clusters
customers by annual income and spending score so we can see which groups
are worth targeting.
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "Mall_Customers.csv"
OUT_DIR = HERE / "viz"
OUT_DIR.mkdir(exist_ok=True)

RED = "#c0392b"
DARK_RED = "#7b241c"
GREY = "#7f8c8d"
LIGHT_GREY = "#bdc3c7"
BG = "#fdfdfd"

plt.rcParams["figure.facecolor"] = BG
plt.rcParams["axes.facecolor"] = BG
plt.rcParams["font.size"] = 10


def scatter_by_gender(ax, df, x, y, title):
    """Scatter plot split by gender, male/female colored differently."""
    male = df[df["Gender"] == "Male"]
    female = df[df["Gender"] == "Female"]
    ax.scatter(male[x], male[y], c=GREY, label="Male", alpha=0.7, edgecolors="white")
    ax.scatter(female[x], female[y], c=RED, label="Female", alpha=0.7, edgecolors="white")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend()


def line_with_fill(ax, x, y, title, xlabel, ylabel):
    """Line plot with a light fill underneath, used for the elbow/silhouette charts."""
    ax.plot(x, y, marker="o", color=RED, linewidth=2)
    ax.fill_between(x, y, min(y), color=RED, alpha=0.08)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={
        "Annual Income (k$)": "Income",
        "Spending Score (1-100)": "SpendingScore",
    })
    return df


def data_quality_check(df):
    print("Rows, columns:", df.shape)
    nulls = df.isnull().sum().sum()
    dupes = df.duplicated().sum()
    print("Missing values:", nulls)
    print("Duplicate rows:", dupes)
    if nulls == 0 and dupes == 0:
        print("Data looks clean, nothing to fix.")


def descriptive_stats(df):
    print("\nAge:", df["Age"].min(), "-", df["Age"].max(), "| mean", round(df["Age"].mean(), 1))
    print("Income (k$):", df["Income"].min(), "-", df["Income"].max(), "| mean", round(df["Income"].mean(), 1))
    print("Spending score:", df["SpendingScore"].min(), "-", df["SpendingScore"].max(), "| mean", round(df["SpendingScore"].mean(), 1))
    print("\nGender split:")
    print(df["Gender"].value_counts())


def plot_eda(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(df["Age"], bins=15, color=RED, edgecolor="white")
    axes[0, 0].set_title("Age distribution")
    axes[0, 0].set_xlabel("Age")

    axes[0, 1].hist(df["Income"], bins=15, color=DARK_RED, edgecolor="white")
    axes[0, 1].set_title("Income distribution")
    axes[0, 1].set_xlabel("Income (k$)")

    axes[1, 0].hist(df["SpendingScore"], bins=15, color=GREY, edgecolor="white")
    axes[1, 0].set_title("Spending score distribution")
    axes[1, 0].set_xlabel("Spending score")

    scatter_by_gender(axes[1, 1], df, "Income", "SpendingScore", "Income vs spending score")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_eda.png", dpi=150)
    plt.close(fig)


def age_group_analysis(df):
    bins = [17, 25, 35, 45, 60, 100]
    labels = ["18-25", "26-35", "36-45", "46-60", "60+"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)

    grouped = df.groupby("AgeGroup", observed=True)["SpendingScore"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(grouped.index.astype(str), grouped.values, color=RED, edgecolor="white")
    ax.set_title("Average spending score by age group")
    ax.set_xlabel("Age group")
    ax.set_ylabel("Avg spending score")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_age_groups.png", dpi=150)
    plt.close(fig)

    return grouped


def pick_k(X_scaled, k_range=range(2, 11)):
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    line_with_fill(axes[0], list(k_range), inertias, "Elbow method", "k", "Inertia")
    line_with_fill(axes[1], list(k_range), silhouettes, "Silhouette score", "k", "Score")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_elbow_silhouette.png", dpi=150)
    plt.close(fig)

    return inertias, silhouettes


def label_clusters(df, cluster_col="Cluster"):
    """
    Give each cluster a human-readable label based on its average income
    and spending score, relative to the dataset median.

    With k=5 there are only four quadrant labels, so one cluster would
    always collide. The cluster whose centroid sits closest to the grand
    centroid (mean of all centroids) is labelled "Middle income, average
    spend" before the quadrant logic runs, avoiding the duplicate.
    """
    summary = df.groupby(cluster_col)[["Income", "SpendingScore"]].mean()
    income_mid = df["Income"].median()
    spend_mid = df["SpendingScore"].median()

    grand = summary.mean()
    distances = ((summary - grand) ** 2).sum(axis=1) ** 0.5
    middle_id = distances.idxmin()

    labels = {}
    for cluster_id, row in summary.iterrows():
        if cluster_id == middle_id:
            labels[cluster_id] = "Middle income, average spend"
            continue
        high_income = row["Income"] >= income_mid
        high_spend = row["SpendingScore"] >= spend_mid
        if high_income and high_spend:
            labels[cluster_id] = "High income, high spend (target)"
        elif high_income and not high_spend:
            labels[cluster_id] = "High income, low spend (re-engage)"
        elif not high_income and high_spend:
            labels[cluster_id] = "Low income, high spend"
        else:
            labels[cluster_id] = "Low income, low spend"

    return labels


def plot_clusters(df, cluster_col, labels, predict_fn):
    """
    predict_fn takes an array of raw [Income, SpendingScore] pairs and returns
    predicted cluster ids. Used to shade the background by predicted cluster
    so the boundaries between segments are visible, not just the points.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [RED, DARK_RED, GREY, LIGHT_GREY, "#e74c3c", "#34495e"]
    color_rgb = np.array([mcolors.to_rgb(c) for c in colors])

    x_min, x_max = df["Income"].min() - 5, df["Income"].max() + 5
    y_min, y_max = df["SpendingScore"].min() - 5, df["SpendingScore"].max() + 5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1.0), np.arange(y_min, y_max, 1.0))
    grid_labels = predict_fn(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    background = color_rgb[grid_labels % len(colors)]

    ax.imshow(
        background, extent=(x_min, x_max, y_min, y_max),
        origin="lower", aspect="auto", alpha=0.15,
    )

    for cluster_id, group in df.groupby(cluster_col):
        color = colors[int(cluster_id) % len(colors)]
        ax.scatter(
            group["Income"], group["SpendingScore"],
            label=labels.get(cluster_id, f"Cluster {cluster_id}"),
            color=color, alpha=0.85, edgecolors="white", s=60,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Annual income (k$)")
    ax.set_ylabel("Spending score")
    ax.set_title("Customer segments")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_clusters.png", dpi=150)
    plt.close(fig)


def plot_gender_by_cluster(df, cluster_col):
    ct = pd.crosstab(df[cluster_col], df["Gender"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ct.plot(kind="bar", ax=ax, color=[RED, GREY], edgecolor="white")
    ax.set_title("Gender split within each cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    ax.legend(title="Gender")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_gender_cluster.png", dpi=150)
    plt.close(fig)


def print_summary(df, cluster_col, labels):
    print("\nCluster summary:")
    for cluster_id, label in labels.items():
        size = (df[cluster_col] == cluster_id).sum()
        pct = size / len(df) * 100
        print(f"  Cluster {cluster_id} ({label}): {size} customers ({pct:.1f}%)")


def main():
    df = load_data()
    data_quality_check(df)
    descriptive_stats(df)
    plot_eda(df)
    age_group_analysis(df)

    features = df[["Income", "SpendingScore"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    inertias, silhouettes = pick_k(X_scaled)

    # k=5 keeps showing up as the elbow point and it's easier to action on
    # than splitting into more segments, so going with that.
    k = 5
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = km.fit_predict(X_scaled)

    labels = label_clusters(df)
    plot_clusters(df, "Cluster", labels, lambda X: km.predict(scaler.transform(X)))
    plot_gender_by_cluster(df, "Cluster")
    print_summary(df, "Cluster", labels)


if __name__ == "__main__":
    main()
