"""
Clustering para segmentación energética (solo países ISO-3) + renewables_share

Dataset esperado:
  - co2_energy_combined_clean.csv

Qué hace:
  1) Carga el dataset
  2) Filtra SOLO países (iso_code válido de 3 letras)
  3) Selecciona un periodo reciente (default 2015–2024)
  4) Feature engineering:
        - fossil_consumption = coal + oil + gas
        - renewables_share = renewables / (renewables + fossil)
  5) Agrega por país (promedio del periodo)
  6) Estandariza variables
  7) Elige k óptimo con silhouette (y guarda gráfico elbow+silhouette)
  8) Aplica K-Means
  9) PCA 2D para visualización y export

Ejecutar:
  python '4. modelo_clustering.py' --csv Data\outputs\co2_energy_combined_clean.csv
  python '4. modelo_clustering.py' --csv Data\outputs\co2_energy_combined_clean.csv --start_year 2010 --end_year 2024 --max_k 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# -----------------------------
# Defaults
# -----------------------------
DEFAULT_START_YEAR = 2015
DEFAULT_END_YEAR = 2024
DEFAULT_MAX_K = 8
RANDOM_STATE = 42

# Variables para clustering (estructurales, comparables)
FEATURES = [
    "energy_per_capita",
    "carbon_intensity_raw",
    "electricity_share_energy",
    "renewables_share",
    "co2_per_capita",
    "energy_per_gdp",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Ruta a co2_energy_combined_clean.csv")
    p.add_argument("--start_year", type=int, default=DEFAULT_START_YEAR)
    p.add_argument("--end_year", type=int, default=DEFAULT_END_YEAR)
    p.add_argument("--max_k", type=int, default=DEFAULT_MAX_K, help="Evalúa k=2..max_k")
    p.add_argument("--outdir", default="outputs")
    return p.parse_args()


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df


def filter_iso3_countries(df: pd.DataFrame) -> pd.DataFrame:
    # SOLO países con ISO-3 (3 letras). Excluye agregados tipo Europe/World/OWID_*
    df = df[df["iso_code"].notna()].copy()
    df["iso_code"] = df["iso_code"].astype(str).str.strip()
    df = df[df["iso_code"].str.len() == 3]
    df = df[df["iso_code"].str.match(r"^[A-Z]{3}$", na=False)]
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Consumo fósil
    for c in ["coal_consumption", "oil_consumption", "gas_consumption", "renewables_consumption"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["fossil_consumption"] = (
        df["coal_consumption"].fillna(0)
        + df["oil_consumption"].fillna(0)
        + df["gas_consumption"].fillna(0)
    )

    # Participación renovables (estructura del mix; evita sesgo por tamaño)
    denom = (df["renewables_consumption"].fillna(0) + df["fossil_consumption"].fillna(0))
    df["renewables_share"] = np.where(denom > 0, df["renewables_consumption"] / denom, np.nan)

    df["renewables_share"] = df["renewables_share"].replace([np.inf, -np.inf], np.nan)

    return df


def aggregate_country_profiles(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    # Filtrar periodo
    dfp = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()

    # Asegurar columnas numéricas relevantes
    for col in FEATURES:
        if col not in dfp.columns:
            raise ValueError(f"Falta la columna requerida para clustering: {col}")
        dfp[col] = pd.to_numeric(dfp[col], errors="coerce")

    # Perfil promedio por país en el periodo
    agg = (
        dfp.groupby(["iso_code", "country"], as_index=False)
           .agg({col: "mean" for col in FEATURES})
    )

    # Quitar filas con NA en variables clave
    agg = agg.dropna(subset=FEATURES).reset_index(drop=True)

    if len(agg) < 20:
        raise ValueError(
            f"Quedaron muy pocos países para clustering (n={len(agg)}). "
            "Revisa años, columnas o calidad de datos."
        )

    return agg


def evaluate_k(X_scaled: np.ndarray, max_k: int) -> tuple[list[float], list[float], int]:
    inertias = []
    silhouettes = []
    ks = list(range(2, max_k + 1))

    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    best_k = ks[int(np.argmax(silhouettes))]
    return inertias, silhouettes, best_k


def plot_k_diagnostics(inertias: list[float], silhouettes: list[float], max_k: int, out_png: Path):
    ks = list(range(2, max_k + 1))

    fig = plt.figure(figsize=(11, 4))

    # Elbow (inertia)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(ks, inertias, marker="o")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inercia (SSE)")
    ax1.set_title("Elbow (K-Means)")

    # Silhouette
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(ks, silhouettes, marker="o")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette")
    ax2.set_title("Silhouette por k")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_clusters_pca(df_out: pd.DataFrame, out_png: Path):
    plt.figure(figsize=(10, 6))
    for c in sorted(df_out["cluster"].unique()):
        sub = df_out[df_out["cluster"] == c]
        plt.scatter(sub["PC1"], sub["PC2"], alpha=0.75, label=f"Cluster {c}")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Clustering de perfiles energéticos (solo países ISO-3) — PCA + K-Means")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(Path(args.csv))
    df = filter_iso3_countries(df)
    df = build_features(df)

    profiles = aggregate_country_profiles(df, args.start_year, args.end_year)

    # Escalado
    X = profiles[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elegir k
    inertias, silhouettes, best_k = evaluate_k(X_scaled, args.max_k)

    diag_png = outdir / "k_diagnostics_elbow_silhouette.png"
    plot_k_diagnostics(inertias, silhouettes, args.max_k, diag_png)

    # KMeans final
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    profiles["cluster"] = kmeans.fit_predict(X_scaled)

    # PCA para visualización (2D)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    profiles["PC1"] = coords[:, 0]
    profiles["PC2"] = coords[:, 1]

    # Export CSV
    out_csv = outdir / "clusters_energy_countries_iso3.csv"
    profiles.to_csv(out_csv, index=False, encoding="utf-8")

    # Plot clusters
    out_png = outdir / "clusters_energy_countries_iso3.png"
    plot_clusters_pca(profiles, out_png)

    # Consola resumen
    print("\n===============================")
    print(" CLUSTERING ENERGÉTICO (SOLO iso_code)")
    print("===============================")
    print(f"Periodo: {args.start_year}-{args.end_year}")
    print(f"Países incluidos: {len(profiles)}")
    print(f"k óptimo (silhouette): {best_k}")
    print("\nSilhouette por k:")
    for i, k in enumerate(range(2, args.max_k + 1)):
        print(f"  k={k}: {silhouettes[i]:.4f}")

    print("\nSalidas generadas:")
    print(f" - CSV: {out_csv.resolve()}")
    print(f" - PNG clusters: {out_png.resolve()}")
    print(f" - PNG diagnóstico k: {diag_png.resolve()}")


if __name__ == "__main__":
    main()
