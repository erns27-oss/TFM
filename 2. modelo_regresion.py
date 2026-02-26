"""
Modelo de regresión para proyección de consumo energético (CO2 + Energy)

Dataset esperado:
  - co2_energy_combined_clean.csv

Qué hace:
  1) Carga el dataset
  2) Selecciona ámbito:
        - global (agregado por año), o
        - una región/país (country == "Europe", "Africa", "Colombia", etc.)
  3) Entrena un modelo LinearRegression para predecir primary_energy_consumption
  4) Proyecta hasta un año futuro (default 2035) usando un escenario sencillo
  5) Exporta:
        - outputs/proyeccion_consumo.csv
        - outputs/proyeccion_consumo.png

Ejecutar (ejemplos):
  python '2. modelo_regresion.py' --csv Data\outputs\co2_energy_combined_clean.csv --scope global
  python '2. modelo_regresion.py' --csv Data\outputs\co2_energy_combined_clean.csv --scope entity --entity Europe
  python '2. modelo_regresion.py' --csv Data\outputs\co2_energy_combined_clean.csv --scope entity --entity Africa --end_year 2040
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -----------------------------
# Config por defecto (escenario)
# -----------------------------
DEFAULT_END_YEAR = 2035

# Supuestos de escenario (puedes cambiarlos)
GROWTH_ELECTRICITY = 0.015   # +1.5% anual
GROWTH_RENEWABLES  = 0.03    # +3.0% anual
DECLINE_CARBON_INT = 0.01    # -1.0% anual
HOLD_ENERGY_PC     = True    # energy_per_capita constante


FEATURES = [
    "year",
    "electricity_generation",
    "renewables_consumption",
    "energy_per_capita",
    "carbon_intensity_raw",
]
TARGET = "primary_energy_consumption"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="Ruta al CSV co2_energy_combined_clean.csv")
    p.add_argument("--scope", type=str, choices=["global", "entity"], default="global",
                   help="global = agrega por year; entity = filtra por country")
    p.add_argument("--entity", type=str, default=None, help='Valor exacto de country (p.ej. "Europe", "Africa")')
    p.add_argument("--end_year", type=int, default=DEFAULT_END_YEAR, help="Año final de proyección")
    p.add_argument("--outdir", type=str, default="outputs", help="Carpeta de salida")
    return p.parse_args()


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Asegurar tipos básicos
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df


def select_scope(df: pd.DataFrame, scope: str, entity: str | None) -> pd.DataFrame:
    if scope == "global":
        df_model = (
            df.groupby("year", as_index=False)
              .agg({
                  TARGET: "sum",
                  "electricity_generation": "sum",
                  "renewables_consumption": "sum",
                  "energy_per_capita": "mean",
                  "carbon_intensity_raw": "mean",
              })
        )
        return df_model

    # scope == "entity"
    if not entity:
        raise ValueError('Si usas --scope entity debes pasar --entity, ej: --entity "Europe"')
    df_entity = df[df["country"] == entity].copy()
    if df_entity.empty:
        # ayuda rápida: mostrar opciones cercanas
        options = sorted(df["country"].dropna().unique().tolist())
        raise ValueError(
            f'No encontré filas con country == "{entity}". '
            f"Ejemplos disponibles: {options[:20]} ..."
        )
    return df_entity


def clean_for_model(df_model: pd.DataFrame) -> pd.DataFrame:
    # Mantener solo columnas necesarias
    needed = FEATURES + [TARGET]
    missing = [c for c in needed if c not in df_model.columns]
    if missing:
        raise ValueError(f"Faltan columnas necesarias para el modelo: {missing}")

    dfc = df_model[needed].copy()

    # Convertir a numérico donde aplique
    for c in needed:
        dfc[c] = pd.to_numeric(dfc[c], errors="coerce")

    # Drop NA
    dfc = dfc.dropna().sort_values("year")

    # Chequeo mínimo
    if len(dfc) < 10:
        raise ValueError(
            f"Quedaron muy pocos datos para entrenar (n={len(dfc)}). "
            "Revisa NA/limpieza o el entity elegido."
        )
    return dfc


def train_model(dfc: pd.DataFrame) -> tuple[LinearRegression, pd.DataFrame, pd.Series]:
    X = dfc[FEATURES]
    y = dfc[TARGET]

    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict(X)

    metrics = pd.Series({
        "R2_in_sample": model.score(X, y),
        "MAE": mean_absolute_error(y, preds),
        "MSE": mean_squared_error(y, preds),
        "RMSE": float(np.sqrt(mean_squared_error(y, preds))),
        "n_obs": len(dfc),
        "year_min": int(dfc["year"].min()),
        "year_max": int(dfc["year"].max()),
    })

    coefs = pd.Series(model.coef_, index=FEATURES).sort_values(ascending=False)
    return model, X.assign(y=y, y_hat=preds), metrics, coefs


def build_future_scenario(dfc: pd.DataFrame, end_year: int) -> pd.DataFrame:
    last = dfc.sort_values("year").iloc[-1]
    start_year = int(last["year"]) + 1
    if end_year < start_year:
        raise ValueError(f"--end_year ({end_year}) debe ser >= {start_year}")

    future_years = np.arange(start_year, end_year + 1)
    t = np.arange(len(future_years))

    energy_pc = np.full_like(future_years, last["energy_per_capita"], dtype=float) if HOLD_ENERGY_PC else \
                last["energy_per_capita"] * (1.0 ** t)

    future = pd.DataFrame({
        "year": future_years.astype(float),
        "electricity_generation": last["electricity_generation"] * ((1 + GROWTH_ELECTRICITY) ** t),
        "renewables_consumption": last["renewables_consumption"] * ((1 + GROWTH_RENEWABLES) ** t),
        "energy_per_capita": energy_pc,
        "carbon_intensity_raw": last["carbon_intensity_raw"] * ((1 - DECLINE_CARBON_INT) ** t),
    })
    return future


def plot_series(dfc: pd.DataFrame, future: pd.DataFrame, y_future: np.ndarray, title: str, out_png: Path):
    plt.figure(figsize=(11, 5))
    plt.plot(dfc["year"], dfc[TARGET], label="Histórico")
    plt.plot(future["year"], y_future, linestyle="--", label="Proyección")
    plt.xlabel("Año")
    plt.ylabel(TARGET)
    plt.title(title)
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
    df_model = select_scope(df, args.scope, args.entity)
    dfc = clean_for_model(df_model)

    model, fitted_df, metrics, coefs = train_model(dfc)

    # Escenario futuro + predicción
    future = build_future_scenario(dfc, args.end_year)
    y_future = model.predict(future[FEATURES])

    # Export de resultados
    out_csv = outdir / "proyeccion_consumo.csv"
    out_png = outdir / "proyeccion_consumo.png"

    results = pd.concat(
        [
            dfc[["year", TARGET]].assign(tipo="historico"),
            future[["year"]].assign(**{TARGET: y_future}, tipo="proyeccion"),
        ],
        ignore_index=True
    ).sort_values(["tipo", "year"])

    results.to_csv(out_csv, index=False, encoding="utf-8")

    scope_label = "GLOBAL" if args.scope == "global" else f'ENTITY: {args.entity}'
    plot_title = f"Proyección de {TARGET} ({scope_label})"
    plot_series(dfc, future, y_future, plot_title, out_png)

    # Consola: resumen
    print("\n=== RESUMEN DEL MODELO ===")
    print(f"Scope: {args.scope}" + (f" | Entity: {args.entity}" if args.entity else ""))
    print("\nMétricas:")
    print(metrics.to_string())

    print("\nCoeficientes (mayor a menor):")
    print(coefs.to_string())

    print("\nSalidas generadas:")
    print(f" - CSV: {out_csv.resolve()}")
    print(f" - PNG: {out_png.resolve()}")


if __name__ == "__main__":
    main()
