"""Solution 3 v2: Hybrid decision engine combining ML probability with attractor resonance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared_dataset import FEATURE_NAMES, IDEAL_VECTOR, generate_projects


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class ResonantAttractor:
    target_vector: np.ndarray
    feature_names: list[str]
    weights: np.ndarray

    def weighted_distance(self, vector: np.ndarray) -> float:
        diff = vector - self.target_vector
        return float(np.sqrt(np.sum(self.weights * np.square(diff))))

    def resonance(self, vector: np.ndarray) -> float:
        return 1.0 / (1.0 + self.weighted_distance(vector))


def build_ml_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=2, random_state=42)),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=2, random_state=42)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=5,
                        min_samples_leaf=3,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def minmax(series: pd.Series) -> pd.Series:
    span = series.max() - series.min()
    if span == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / span


def save_hybrid_plot(frame: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(frame["ml_probability"], frame["resonance"], alpha=0.7)
    plt.xlabel("ML probability")
    plt.ylabel("Attractor resonance")
    plt.title("Hybrid decision map: ML probability vs resonance")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    dataset = generate_projects(n_samples=160, seed=42)
    df = dataset.frame

    X = df[FEATURE_NAMES]
    y = df["class_label"]

    _, X_test, _, y_test, _, test_index = train_test_split(
        X,
        y,
        df.index,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model_probs: dict[str, np.ndarray] = {}
    metric_rows = []
    for name, model in build_ml_models().items():
        X_train, X_eval, y_train, y_eval = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_eval)[:, 1]
        model_probs[name] = proba
        metric_rows.append({"model": name, "roc_auc": roc_auc_score(y_eval, proba)})

    ensemble_probability = (model_probs["logistic_regression"] + model_probs["random_forest"]) / 2.0

    attractor = ResonantAttractor(
        target_vector=IDEAL_VECTOR.copy(),
        feature_names=FEATURE_NAMES.copy(),
        weights=np.array([1.25, 1.00, 1.10, 1.50, 1.40], dtype=float),
    )

    resonance_scores = X_test.apply(
        lambda row: attractor.resonance(row.to_numpy(dtype=float)),
        axis=1,
    )

    hybrid_df = pd.DataFrame(
        {
            "project_name": df.loc[test_index, "project_name"].values,
            "true_label": y_test.values,
            "ml_probability": ensemble_probability,
            "resonance": resonance_scores.values,
        }
    )
    hybrid_df["ml_probability_scaled"] = minmax(hybrid_df["ml_probability"])
    hybrid_df["resonance_scaled"] = minmax(hybrid_df["resonance"])
    hybrid_df["hybrid_score"] = 0.60 * hybrid_df["ml_probability_scaled"] + 0.40 * hybrid_df["resonance_scaled"]
    hybrid_df = hybrid_df.sort_values(by="hybrid_score", ascending=False)

    top10 = hybrid_df.head(10).copy()
    top10["decision_bucket"] = pd.cut(
        top10["hybrid_score"],
        bins=[-0.01, 0.45, 0.65, 1.0],
        labels=["reject", "review", "prioritize"],
    )

    hybrid_csv = OUTPUT_DIR / "solution_3_v2_hybrid_scores.csv"
    metrics_csv = OUTPUT_DIR / "solution_3_v2_ml_metrics.csv"
    plot_png = OUTPUT_DIR / "solution_3_v2_hybrid_plot.png"
    top10_csv = OUTPUT_DIR / "solution_3_v2_top10.csv"

    hybrid_df.to_csv(hybrid_csv, index=False)
    pd.DataFrame(metric_rows).to_csv(metrics_csv, index=False)
    top10.to_csv(top10_csv, index=False)
    save_hybrid_plot(hybrid_df, plot_png)

    print("=" * 72)
    print("SOLUTION 3 v2 | HYBRID ML + ATTRACTOR")
    print("=" * 72)
    print("ML model quality:")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print("\nTop 10 hybrid decisions:")
    print(top10[["project_name", "true_label", "ml_probability", "resonance", "hybrid_score", "decision_bucket"]].to_string(index=False))
    print("\nSaved files:")
    for path in [hybrid_csv, metrics_csv, plot_png, top10_csv]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
