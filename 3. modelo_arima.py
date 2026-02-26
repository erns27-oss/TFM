"""
Series temporales (ARIMA) para proyección energética: Europe vs South America

Dataset esperado:
  - co2_energy_combined_clean.csv

Qué hace:
  1) Carga el dataset
  2) Filtra dos regiones por country: "Europe" y "South America"
  3) Construye serie anual de primary_energy_consumption por región (1985–2023 en tu caso)
  4) Ajusta ARIMA (con búsqueda simple por AIC en una grilla pequeña)
  5) Proyecta N años hacia adelante con intervalos de confianza
  6) Exporta CSV + PNG comparativo

Ejecutar:
  python '3. modelo_arima.py' --csv Data\outputs\co2_energy_combined_clean.csv --horizon 12 --outdir outputs

Notas:
- Este enfoque NO usa variables exógenas: proyecta a partir del patrón histórico.
- Ideal para complementar tu regresión (explicativa) con una proyección dinámica.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings


# -----------------------------
# CONFIG
# -----------------------------
REGIONS = ["Europe", "South America"]
TARGET = "primary_energy_consumption"

# Grilla pequeña para elegir (p,d,q) por AIC.
# d en [0,1,2] porque series con tendencia suelen requerir diferenciación.
P_VALUES = [0, 1, 2]
D_VALUES = [0, 1, 2]
Q_VALUES = [0, 1, 2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="Ruta al CSV co2_energy_combined_clean.csv")
    p.add_argument("--horizon", type=int, default=12, help="Años a proyectar hacia adelante (ej. 12)")
    p.add_argument("--outdir", type=str, default="outputs", help="Carpeta de salida")
    p.add_argument("--ci", type=float, default=0.95, help="Nivel intervalo de confianza (0.90, 0.95, 0.99)")
    return p.parse_args()


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df


def build_series(df: pd.DataFrame, region: str) -> pd.Series:
    dfr = df[df["country"] == region].copy()
    if dfr.empty:
        available = sorted(df["country"].dropna().unique().tolist())
        raise ValueError(
            f'No encontré "{region}" en country. '
            f"Ejemplos disponibles: {available[:30]} ..."
        )

    # Asegurar numérico
    dfr[TARGET] = pd.to_numeric(dfr[TARGET], errors="coerce")

    # Serie anual (ya suele venir anual, pero lo dejamos robusto)
    s = (
        dfr.groupby("year")[TARGET]
           .sum(min_count=1)
           .dropna()
           .sort_index()
    )

    # Quitar años inválidos
    s = s[~s.index.isna()]

    if len(s) < 12:
        raise ValueError(f"Serie muy corta para {region} (n={len(s)}). Revisa datos/filtros.")

    # Convertir índice a int
    # s.index = s.index.astype(int)
    years_int = s.index.astype(int)
    s.index = pd.to_datetime(years_int, format="%Y")
    return s


def select_arima_order(series: pd.Series) -> tuple[int, int, int]:
    """
    Búsqueda simple por AIC en una grilla pequeña.
    Devuelve el (p,d,q) con menor AIC.
    """
    best_order = None
    best_aic = np.inf

    # Evitar ruido de warnings por convergencia
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    y = series.astype(float)

    for p in P_VALUES:
        for d in D_VALUES:
            for q in Q_VALUES:
                # (0,0,0) suele ser demasiado trivial; lo permitimos, pero rara vez gana
                try:
                    model = ARIMA(y, order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False)
                    res = model.fit()
                    aic = res.aic
                    if np.isfinite(aic) and aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                except Exception:
                    continue

    if best_order is None:
        # fallback razonable
        best_order = (1, 1, 1)

    return best_order


def fit_and_forecast(series: pd.Series, horizon: int, ci: float) -> tuple[pd.DataFrame, tuple[int,int,int], float]:
    order = select_arima_order(series)

    y = series.astype(float)
    model = ARIMA(y, order=order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit()

    # Forecast horizon
    fc = res.get_forecast(steps=horizon)
    mean = fc.predicted_mean
    conf = fc.conf_int(alpha=1 - ci)  # columns: lower/upper

    # Años futuros
    # last_year = int(series.index.max())
    # future_years = np.arange(last_year + 1, last_year + horizon + 1)
    last_dt = pd.to_datetime(series.index.max())
    future_index = pd.date_range(
        start=last_dt + pd.DateOffset(years=1),
        periods=horizon,
        freq="YS"
    )

    out = pd.DataFrame({
        "date": future_index,
        "year": future_index.year.astype(int),
        "y_pred": mean.values,
        "y_lower": conf.iloc[:, 0].values,
        "y_upper": conf.iloc[:, 1].values,
    })

    return out, order, res.aic


def to_long_table(series: pd.Series, forecast_df: pd.DataFrame, region: str) -> pd.DataFrame:
    hist = pd.DataFrame({
        "year": series.index.astype(int),
        "region": region,
        "tipo": "historico",
        "value": series.values.astype(float),
        "lower": np.nan,
        "upper": np.nan,
    })

    proj = pd.DataFrame({
        "year": forecast_df["year"].astype(int),
        "region": region,
        "tipo": "proyeccion",
        "value": forecast_df["y_pred"].astype(float),
        "lower": forecast_df["y_lower"].astype(float),
        "upper": forecast_df["y_upper"].astype(float),
    })

    return pd.concat([hist, proj], ignore_index=True)


def plot_comparison(series_by_region: dict[str, pd.Series],
                    forecast_by_region: dict[str, pd.DataFrame],
                    ci: float,
                    out_png: Path):
    plt.figure(figsize=(12, 6))

    # Histórico
    for region, s in series_by_region.items():
        plt.plot(s.index, s.values, label=f"{region} (hist)")

    # Proyección + bandas
    for region, f in forecast_by_region.items():
        plt.plot(f["year"], f["y_pred"], linestyle="--", label=f"{region} (proj)")
        plt.fill_between(
            f["year"],
            f["y_lower"],
            f["y_upper"],
            alpha=0.2
        )

    plt.xlabel("Año")
    plt.ylabel(TARGET)
    plt.title(f"Proyección por series temporales (ARIMA) – Europe vs South America (CI {int(ci*100)}%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)

    # Construir series
    series_by_region: dict[str, pd.Series] = {}
    forecast_by_region: dict[str, pd.DataFrame] = {}
    rows = []

    print("\n===============================")
    print(" SERIES TEMPORALES: ARIMA (Europe vs South America)")
    print("===============================")
    print(f"Horizonte: {args.horizon} años | CI: {args.ci}\n")

    for region in REGIONS:
        s = build_series(df, region)
        f, order, aic = fit_and_forecast(s, args.horizon, args.ci)

        series_by_region[region] = s
        forecast_by_region[region] = f

        rows.append(to_long_table(s, f, region))

        print(f"--- {region} ---")
        print(f"Observaciones: {len(s)} | Años: {s.index.min().year}-{s.index.max().year}")
        print(f"Mejor ARIMA(p,d,q): {order} | AIC: {aic:.2f}\n")

    # Export CSV (formato largo para Power BI)
    out_csv = outdir / "ts_forecast_europe_vs_south_america.csv"
    results = pd.concat(rows, ignore_index=True).sort_values(["region", "tipo", "year"])
    results.to_csv(out_csv, index=False, encoding="utf-8")

    # Export PNG
    out_png = outdir / "ts_forecast_europe_vs_south_america.png"
    plot_comparison(series_by_region, forecast_by_region, args.ci, out_png)

    print("===============================")
    print(" Salidas generadas")
    print("===============================")
    print(f"CSV: {out_csv.resolve()}")
    print(f"PNG: {out_png.resolve()}")


if __name__ == "__main__":
    main()
