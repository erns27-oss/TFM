"""
EDA combinado CO2 + Energía (OWID) 
(limpia CO2 primero y luego hace merge)
Archivos esperados:
    - owid-co2-data.csv
    - owid-energy-data.xlsx
Qué hace:
    1. Carga ambos datasets
    2. Limpia primero CO2: elimina filas con >60% faltantes (NaN o 0) (sin contar llaves)
    3. Merge por (country, year)
    4. Reconcilia iso_code (si existe) como atributo
    5. Reconcilia población y PIB
    6. Limpia de nuevo el dataset combinado con el mismo umbral
    7. EDA básico por consola
    8. Creación tablas agregadas para el TFM
    9. Generación gráficas preliminares en PNG
Ejecutar:
    python '1. data_cleaning_eda.py'
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------
RUTA_CO2 = Path("Data/owid-co2-data.csv")
RUTA_ENERGY = Path("Data/owid-energy-data.csv")
RUTA_RENEW_LONG = Path("Data/output_renewable_energy/renewable_energy_long_clean.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
ENERGY_MAIN_COLS = [
    "primary_energy_consumption",
    "coal_consumption",
    "oil_consumption",
    "gas_consumption",
    "renewables_consumption",
    "nuclear_consumption",
    "solar_consumption",
    "wind_consumption",
    "hydro_consumption",
    "low_carbon_consumption",
    "electricity_generation",
    "electricity_share_energy",
    "co2_electricity_estimated"
]
MERGE_KEYS = ["country", "year"] 
# Columnas "calculadas" / derivadas del bloque de energía
ENERGY_CALC_COLS = [
    "energy_per_capita",
    "energy_per_gdp",
]
# CO2 mínimas que queremos conservar del dataset CO2
CO2_MIN_COLS = [
    "co2",
    "co2_per_capita",
]
def recortar_co2_para_merge(df_co2: pd.DataFrame) -> pd.DataFrame:
    """
    Deja en CO2 únicamente:
    - llaves (country, year)
    - iso_code (si existe)
    - CO2 mínimas (co2, co2_per_capita)
    - energía main + calculadas (si existieran en CO2)
    """
    keep = list(MERGE_KEYS)
    if "iso_code" in df_co2.columns:
        keep.append("iso_code")
    keep += [c for c in CO2_MIN_COLS if c in df_co2.columns]
    keep += [c for c in (ENERGY_MAIN_COLS + ENERGY_CALC_COLS) if c in df_co2.columns]
    # dedupe manteniendo orden
    keep = list(dict.fromkeys(keep))
    return df_co2.loc[:, keep].copy()
# Hacer limpieza despues del merge
LIMPIAR_POST_MERGE = True
# ---------------------------------------------------------------------
# FUNCIONES DE CARGA
# ---------------------------------------------------------------------
def cargar_co2_energy(ruta_co2: Path, ruta_energy: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Cargando datasets...")
    df_co2 = pd.read_csv(ruta_co2)
    df_energy = pd.read_csv(ruta_energy)
    # Asegurar tipo de year consistente
    for df in (df_co2, df_energy):
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
        if "country" in df.columns:
            df["country"] = df["country"].astype(str).str.strip()
    print(f"   CO2 shape:    {df_co2.shape}")
    print(f"   Energy shape: {df_energy.shape}")
    return df_co2, df_energy

def agregar_capacidad_instalada(
    df: pd.DataFrame,
    df_capacidad: pd.DataFrame
) -> pd.DataFrame:
    """
    Agrega capacidad instalada (GW) al dataset combinado
    usando (iso_code, year) como llave.
    """
    print("\n🔌 Agregando capacidad instalada...")
    required_main = {"iso_code", "year"}
    required_cap = {"iso_code", "year", "installed_capacity_gw"}
    if not required_main.issubset(df.columns):
        print("⚠ El dataset combinado no tiene iso_code/year.")
        return df
    if not required_cap.issubset(df_capacidad.columns):
        print("⚠ El dataset de capacidad no tiene columnas requeridas.")
        return df
    out = df.merge(
        df_capacidad,
        on=["iso_code", "year"],
        how="left"
    )
    print("   ➜ Capacidad instalada agregada.")
    print(f"   ➜ Shape después del merge: {out.shape}")
    return out


INDICADOR_CAPACIDAD = "Electricity Installed Capacity"

def construir_capacidad_desde_long_clean(ruta_long: Path) -> pd.DataFrame:
    """
    Lee renewable_energy_long_clean.csv (LONG) y devuelve capacidad instalada país-año:
      iso_code, year, installed_capacity_gw
    Reglas:
      - Indicator = Electricity Installed Capacity
      - Solo países ISO3 válidos (A-Z{3})
      - Excluye World (WLD) y OWID_*
      - Normaliza MW -> GW
      - Agrega por país-año (sum)
    """
    print("\n Construyendo capacidad instalada desde renewable_energy_long_clean.csv...")
    d = pd.read_csv(ruta_long)
    required = {"ISO3", "year", "Indicator", "value"}
    faltan = required - set(d.columns)
    if faltan:
        print(f" Faltan columnas en long_clean: {faltan}")
        return pd.DataFrame()
    # 1) Solo indicador capacidad
    d = d[d["Indicator"] == INDICADOR_CAPACIDAD].copy()
    if d.empty:
        print(f" No hay filas con Indicator='{INDICADOR_CAPACIDAD}'.")
        return pd.DataFrame()
    # 2) Solo países (ISO3 real) y quitar World/agregados
    d["ISO3"] = d["ISO3"].astype(str).str.strip()
    mask_iso3 = d["ISO3"].str.match(r"^[A-Z]{3}$", na=False)
    d = d[mask_iso3 & (d["ISO3"] != "WLD") & ~d["ISO3"].str.startswith("OWID_", na=False)].copy()
    if d.empty:
        print(" No quedaron países válidos tras filtrar ISO3 (sin WLD/OWID_*).")
        return pd.DataFrame()
    # 3) year y value numéricos
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["year", "value"]).copy()
    d["year"] = d["year"].astype(int)
    # 4) Normalizar unidades: MW -> GW
    if "Unit" in d.columns:
        d["Unit"] = d["Unit"].astype(str).str.strip()
        mask_mw = d["Unit"].isin(["Megawatt", "MW"])
        d.loc[mask_mw, "value"] = d.loc[mask_mw, "value"] * 1e-3
    # 5) Agregar país-año 
    cap = (
        d.groupby(["ISO3", "year"], as_index=False)["value"]
         .sum(min_count=1)
         .rename(columns={"ISO3": "iso_code", "value": "installed_capacity_gw"})
    )
    print(f"   ➜ Capacidad instalada país-año: {cap.shape}")
    return cap
# ---------------------------------------------------------------------
# LIMPIEZA (ANTES Y DESPUÉS)
# ---------------------------------------------------------------------
def limpiar_filas_incompletas(
    df: pd.DataFrame,
    umbral: float = 0.6,
    cols_no_contar: list[str] | None = None,
    tratar_ceros_como_na: bool = True,
) -> pd.DataFrame:
    """
    Elimina filas que tengan más del umbral de columnas faltantes.
    - Se considera faltantes: NaN y 0 en columnas numéricas
    - NO cuenta para el porcentaje las columnas en cols_no_contar (ej: country, year)
    - Excluye year de la regla "0 -> NaN" para no romper llaves
    """
    cols_no_contar = cols_no_contar or []
    print(f"\nLimpiando filas con demasiados missing (>{int(umbral*100)}%)...")
    df_work = df.copy()
    # Definir columnas a evaluar (excluyendo llaves/atributos)
    cols_eval = [c for c in df_work.columns if c not in cols_no_contar]
    if not cols_eval:
        print(" No hay columnas para evaluar missing. No se aplica limpieza.")
        return df_work
    # Convertir 0 -> NaN solo en columnas numéricas evaluadas (excluyendo year)
    if tratar_ceros_como_na:
        cols_num_eval = [
            c for c in cols_eval
            if pd.api.types.is_numeric_dtype(df_work[c]) and c != "year"
        ]
        if cols_num_eval:
            df_work[cols_num_eval] = df_work[cols_num_eval].replace(0, np.nan)
    # Porcentaje missing por fila SOLO en cols_eval
    porcentaje_missing = df_work[cols_eval].isna().mean(axis=1)
    before = len(df_work)
    df_filtrado = df_work.loc[porcentaje_missing <= umbral].copy()
    after = len(df_filtrado)
    print(f"   Filas antes:      {before}")
    print(f"   Filas después:    {after}")
    print(f"   Filas eliminadas: {before - after}")
    return df_filtrado
# ---------------------------------------------------------------------
# MEZCLA
# ---------------------------------------------------------------------
def mezclar_datasets(df_co2: pd.DataFrame, df_energy: pd.DataFrame) -> pd.DataFrame:
    """
    Mezcla ambos datasets por (country, year)
    """
    print("\n Mezclando CO2 + Energía ...")
    # ✅ Recortar CO2 antes del merge (solo ENERGY_MAIN_COLS + calculadas + co2 mínimas)
    df_co2 = recortar_co2_para_merge(df_co2)
    # Aseguramos que las llaves no sean NaN
    df_co2 = df_co2.dropna(subset=MERGE_KEYS).copy()
    df_energy = df_energy.dropna(subset=MERGE_KEYS).copy()
    df_merged = df_co2.merge(
        df_energy,
        on=MERGE_KEYS,
        how="outer",
        suffixes=("_co2", "_energy")
    )
    print(f"   dataset combinado (antes de reconciliar columnas): {df_merged.shape}")
    # Reconcilia iso_code como atributo (si existe en ambos)
    if "iso_code_co2" in df_merged.columns or "iso_code_energy" in df_merged.columns:
        df_merged["iso_code"] = df_merged.get("iso_code_co2").fillna(df_merged.get("iso_code_energy"))
        df_merged.drop(columns=[c for c in ["iso_code_co2", "iso_code_energy"] if c in df_merged.columns], inplace=True)
    # Resolver población
    if "population_co2" in df_merged.columns or "population_energy" in df_merged.columns:
        df_merged["population"] = df_merged.get("population_co2").fillna(df_merged.get("population_energy"))
        df_merged.drop(columns=[c for c in ["population_co2", "population_energy"] if c in df_merged.columns], inplace=True)
    # Resolver gdp
    if "gdp_co2" in df_merged.columns or "gdp_energy" in df_merged.columns:
        df_merged["gdp"] = df_merged.get("gdp_co2").fillna(df_merged.get("gdp_energy"))
        df_merged.drop(columns=[c for c in ["gdp_co2", "gdp_energy"] if c in df_merged.columns], inplace=True)
    # Resolver primary_energy_consumption, energy_per_capita, energy_per_gdp, electricity_share_energy, co2_electricity_estimated
    for col_base in ["primary_energy_consumption", "energy_per_capita", "energy_per_gdp", "electricity_share_energy", "co2_electricity_estimated"]:
        col_co2 = f"{col_base}_co2"
        col_energy = f"{col_base}_energy"
        if col_co2 in df_merged.columns or col_energy in df_merged.columns:
            df_merged[col_base] = df_merged.get(col_energy).fillna(df_merged.get(col_co2))
            df_merged.drop(columns=[c for c in [col_co2, col_energy] if c in df_merged.columns], inplace=True)
    print(f"  Dataset combinado (después de reconciliar columnas): {df_merged.shape}")
    return df_merged



# ---------------------------------------------------------------------
# INTENSIDAD DE CARBONO
# ---------------------------------------------------------------------
def calcular_intensidad_carbono(df: pd.DataFrame) -> pd.DataFrame:
    """
    carbon_intensity_raw = co2 / primary_energy_consumption
    """
    print("\n Calculando intensidad de carbono (simple)...")
    if {"co2", "primary_energy_consumption"}.issubset(df.columns):
        df["carbon_intensity_raw"] = df["co2"] / df["primary_energy_consumption"]
        print("   ➜ Columna 'carbon_intensity_raw' creada.")
    else:
        print("   ⚠ No se pudieron encontrar 'co2' y 'primary_energy_consumption'.")
    return df


# ---------------------------------------------------------------------
# CO2 ASOCIADO AL SISTEMA ELÉCTRICO (ESTIMACIÓN)
# ---------------------------------------------------------------------
def calcular_co2_electrico_estimado(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estima las emisiones de CO2 asociadas al sistema eléctrico usando:
        co2_electricity_estimated = co2 * (electricity_share_energy / 100)

    Supuesto:
    - electricity_share_energy está en porcentaje (0–100), estándar OWID.
    - co2 está en MtCO2.
    """
    print("\n Calculando CO₂ asociado al sistema eléctrico (estimado)...")

    required = {"co2", "electricity_share_energy"}
    if not required.issubset(df.columns):
        print("   ⚠ No se encontraron 'co2' y/o 'electricity_share_energy'.")
        return df

    # Asegurar tipo numérico
    df["co2"] = pd.to_numeric(df["co2"], errors="coerce")
    df["electricity_share_energy"] = pd.to_numeric(
        df["electricity_share_energy"], errors="coerce"
    )

    # Cálculo
    df["co2_electricity_estimated"] = (
        df["co2"] * (df["electricity_share_energy"] / 100.0)
    )

    print("   ➜ Columna 'co2_electricity_estimated' creada.")
    return df


# ---------------------------------------------------------------------
# EDA BÁSICO
# ---------------------------------------------------------------------
def eda_basico(df: pd.DataFrame, nombre: str = "df") -> None:
    print("\n" + "=" * 80)
    print(f"EDA BÁSICO :: {nombre}")
    print("=" * 80)
    print("\n * Shape:", df.shape)
    print("\n * Primeras filas:")
    print(df.head())
    print("\n * Info:")
    df.info()
    print("\n * Nulos por columna (top 20):")
    print(df.isna().sum().sort_values(ascending=False).head(20))
    print("\n * Descripción numérica (primeras 10 columnas numéricas):")
    print(df.select_dtypes(include=[np.number]).iloc[:, :10].describe())
def _safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
def resumen_calidad_datos(df: pd.DataFrame, output_dir: Path, nombre: str = "dataset"):
    """
    Reporte de calidad:
    - nulos, ceros (numéricos), cardinalidad
    - filas con %missing alto
    - duplicados por llaves (country, year)
    Guarda CSVs en output_dir.
    """
    print("\n Reporte de calidad de datos...")
    _safe_mkdir(output_dir)
    # ---- Info general
    print(f"   Shape: {df.shape}")
    print(f"   Duplicados totales (filas): {df.duplicated().sum()}")
    # ---- Duplicados por llave
    if {"country", "year"}.issubset(df.columns):
        dup_key = df.duplicated(subset=["country", "year"]).sum()
        print(f"   Duplicados por (country, year): {dup_key}")
    # ---- Nulos por columna
    nulls = df.isna().sum().sort_values(ascending=False)
    nulls_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    rep_nulls = pd.DataFrame({
        "null_count": nulls,
        "null_pct": nulls_pct.round(2),
        "dtype": df.dtypes.astype(str)
    })
    rep_nulls.to_csv(output_dir / f"qa_nulls_{nombre}.csv", index=True)
    print(f"   qa_nulls_{nombre}.csv")
    # ---- Ceros por columna (solo numéricas)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        zeros = (df[num_cols] == 0).sum().sort_values(ascending=False)
        zeros_pct = ((df[num_cols] == 0).mean() * 100).sort_values(ascending=False)
        rep_zeros = pd.DataFrame({"zero_count": zeros, "zero_pct": zeros_pct.round(2)})
        rep_zeros.to_csv(output_dir / f"qa_zeros_{nombre}.csv", index=True)
        print(f"   qa_zeros_{nombre}.csv")
    # ---- Cardinalidad (categóricas / texto)
    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if obj_cols:
        card = df[obj_cols].nunique(dropna=True).sort_values(ascending=False)
        rep_card = pd.DataFrame({"n_unique": card})
        rep_card.to_csv(output_dir / f"qa_cardinality_{nombre}.csv", index=True)
        print(f"   qa_cardinality_{nombre}.csv")
    # ---- % missing por fila (considerando 0 como missing en numéricas)
    df_tmp = df.copy()
    if num_cols:
        df_tmp[num_cols] = df_tmp[num_cols].replace(0, np.nan)
    row_missing_pct = df_tmp.isna().mean(axis=1) * 100
    df_row = pd.DataFrame({"row_missing_pct": row_missing_pct.round(2)})
    df_row["country"] = df["country"] if "country" in df.columns else None
    df_row["year"] = df["year"] if "year" in df.columns else None
    df_row.sort_values("row_missing_pct", ascending=False).head(200).to_csv(
        output_dir / f"qa_top_missing_rows_{nombre}.csv", index=False
    )
    print(f"   qa_top_missing_rows_{nombre}.csv (top 200)")


def outliers_iqr_resumen(df: pd.DataFrame, output_dir: Path, nombre: str = "dataset", top_n: int = 25):
    """
    Resumen de outliers por IQR para columnas numéricas.
    Guarda un CSV con el top de columnas con mayor % de outliers.
    """
    print("\n Resumen de outliers (IQR)...")
    _safe_mkdir(output_dir)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        print("   No hay columnas numéricas para outliers.")
        return
    rows = []
    for c in num_cols:
        s = df[c].dropna()
        if s.empty:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        out = ((s < low) | (s > high)).mean() * 100
        rows.append((c, float(out), float(q1), float(q3), float(low), float(high), int(s.shape[0])))
    if not rows:
        print("   No se pudo calcular outliers (IQR=0 o sin datos).")
        return
    rep = pd.DataFrame(rows, columns=["col", "outlier_pct", "q1", "q3", "low", "high", "n_non_null"])
    rep = rep.sort_values("outlier_pct", ascending=False).head(top_n)
    rep.to_csv(output_dir / f"eda_outliers_iqr_{nombre}.csv", index=False)
    print(f"   ➜ eda_outliers_iqr_{nombre}.csv (top {top_n})")


def plot_missing_by_year(df: pd.DataFrame, output_dir: Path, nombre: str = "dataset"):
    """
    Línea: promedio %missing por año (considera 0 como missing solo en numéricas).
    """
    print("\n Gráfico: missing promedio por año...")
    if "year" not in df.columns:
        print("   No existe 'year'.")
        return
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_tmp = df.copy()
    if num_cols:
        df_tmp[num_cols] = df_tmp[num_cols].replace(0, np.nan)
    miss_pct_row = df_tmp.isna().mean(axis=1) * 100
    tmp = pd.DataFrame({"year": df["year"], "row_missing_pct": miss_pct_row})
    tmp = tmp.dropna(subset=["year"]).groupby("year")["row_missing_pct"].mean().reset_index()
    plt.figure()
    plt.plot(tmp["year"], tmp["row_missing_pct"])
    plt.xlabel("Año")
    plt.ylabel("% missing promedio por fila")
    plt.title("Missing promedio por año")
    plt.tight_layout()
    out_path = output_dir / f"missing_by_year_{nombre}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"   ➜ {out_path.name}")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path, nombre: str = "dataset", max_cols: int = 25):
    """
    Heatmap de correlación para las columnas numéricas más completas.
    """
    print("\n Gráfico: heatmap de correlaciones (numéricas)...")
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        print("   No hay numéricas para correlación.")
        return
    # Seleccionar las columnas con menos missing (más completas)
    completeness = num.notna().mean().sort_values(ascending=False)
    cols = completeness.head(max_cols).index.tolist()
    corr = num[cols].corr()
    plt.figure(figsize=(12, 9))
    plt.imshow(corr, aspect="auto")
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    plt.title("Correlación (Pearson) - Top columnas numéricas")
    plt.colorbar()
    plt.tight_layout()
    out_path = output_dir / f"corr_heatmap_{nombre}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"   ➜ {out_path.name}")


def top_correlaciones_con_objetivo(df: pd.DataFrame, output_dir: Path, objetivo: str = "co2", nombre: str = "dataset"):
    """
    Calcula y guarda top correlaciones con una variable objetivo (ej co2).
    """
    if objetivo not in df.columns:
        print(f"    No existe '{objetivo}'. Se omite top correlaciones.")
        return
    num = df.select_dtypes(include=[np.number]).copy()
    if objetivo not in num.columns:
        print(f"    '{objetivo}' no es numérica. Se omite.")
        return
    corr = num.corr()[objetivo].dropna().sort_values(ascending=False)
    rep = corr.to_frame("corr_with_" + objetivo)
    rep.to_csv(output_dir / f"top_corr_{objetivo}_{nombre}.csv")
    print(f"    top_corr_{objetivo}_{nombre}.csv")


def tabla_top_paises_ultimo_anio(df: pd.DataFrame, output_dir: Path, nombre: str = "dataset"):
    """
    Top países por CO2 y por energía primaria en el último año disponible.
    SOLO países con iso_code válido.
    Guarda 2 CSVs.
    """
    print("\n Tablas: top países (último año)...")
    required_cols = {"country", "year", "iso_code"}
    if not required_cols.issubset(df.columns):
        print("    Faltan columnas requeridas:", required_cols - set(df.columns))
        return
    # Último año válido
    last_year = pd.to_numeric(df["year"], errors="coerce").max()
    if pd.isna(last_year):
        print("    No se pudo identificar el último año.")
        return
    # Filtrar:
    # - último año
    # - iso_code no nulo
    # - iso_code de longitud 3 (estándar ISO-3)
    d = df[
        (df["year"] == last_year) &
        (df["iso_code"].notna()) &
        (df["iso_code"].astype(str).str.len() == 3)
    ].copy()
    if d.empty:
        print("    No hay países válidos con iso_code para el último año.")
        return
    # -----------------------------
    # TOP 25 CO2
    # -----------------------------
    if "co2" in d.columns:
        top_co2 = (
            d[["country", "iso_code", "year", "co2"]]
            .dropna(subset=["co2"])
            .sort_values("co2", ascending=False)
            .head(25)
        )
        top_co2.to_csv(
            output_dir / f"top25_co2_last_year_{nombre}.csv",
            index=False
        )
        print(f"    top25_co2_last_year_{nombre}.csv (year={int(last_year)})")
    # -----------------------------
    # TOP 25 Energía primaria
    # -----------------------------
    if "primary_energy_consumption" in d.columns:
        top_en = (
            d[["country", "iso_code", "year", "primary_energy_consumption"]]
            .dropna(subset=["primary_energy_consumption"])
            .sort_values("primary_energy_consumption", ascending=False)
            .head(25)
        )
        top_en.to_csv(
            output_dir / f"top25_energy_last_year_{nombre}.csv",
            index=False
        )
        print(f"   ➜ top25_energy_last_year_{nombre}.csv (year={int(last_year)})")


def plot_world_timeseries(df: pd.DataFrame, output_dir: Path, nombre: str = "dataset"):
    """
    Series temporales para World: co2, primary_energy_consumption, carbon_intensity_raw.
    """
    print("\n Gráfico: series temporales (World)...")
    if "country" not in df.columns or "year" not in df.columns:
        print("Faltan columnas country/year.")
        return
    w = df[df["country"] == "World"].sort_values("year")
    if w.empty:
        print("No se encontró 'World'.")
        return
    cols = [c for c in ["co2", "primary_energy_consumption", "carbon_intensity_raw"] if c in w.columns]
    if not cols:
        print(" Hay columnas esperadas para graficar.")
        return
    plt.figure()
    for c in cols:
        plt.plot(w["year"], w[c], label=c)
    plt.xlabel("Año")
    plt.ylabel("Valor")
    plt.title("World - CO2 / Energía / Intensidad")
    plt.legend()
    plt.tight_layout()
    out_path = output_dir / f"world_timeseries_{nombre}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"   ➜ {out_path.name}")


def eda_completo_dataset_combinado(df: pd.DataFrame, output_dir: Path, nombre: str = "co2_energy_combined"):
    """
    EDA completo: consola y archivos (CSVs y PNGs) en output_dir.
    """
    print("\n" + "=" * 80)
    print(f" EDA COMPLETO :: {nombre}")
    print("=" * 80)
    _safe_mkdir(output_dir)
    # 1) EDA básico (consola)
    eda_basico(df, nombre=nombre)
    # 2) Calidad
    resumen_calidad_datos(df, output_dir=output_dir, nombre=nombre)
    # 3) Missing por año
    plot_missing_by_year(df, output_dir=output_dir, nombre=nombre)
    # 4) Outliers (IQR)
    outliers_iqr_resumen(df, output_dir=output_dir, nombre=nombre, top_n=25)
    # 5) Correlaciones
    plot_correlation_heatmap(df, output_dir=output_dir, nombre=nombre, max_cols=25)
    top_correlaciones_con_objetivo(df, output_dir=output_dir, objetivo="co2", nombre=nombre)
    # 6) Tablas top países (último año)
    tabla_top_paises_ultimo_anio(df, output_dir=output_dir, nombre=nombre)
    # 7) Series World
    plot_world_timeseries(df, output_dir=output_dir, nombre=nombre)
    print("\n EDA completo generado (consola y CSV/PNG en output_dir).")
# ---------------------------------------------------------------------
# TABLAS AGREGADAS
# ---------------------------------------------------------------------
def columnas_energia_presentes(df: pd.DataFrame):
    return [c for c in ENERGY_MAIN_COLS if c in df.columns]
def tabla_energia_global_anual(df: pd.DataFrame) -> pd.DataFrame:
    cols = columnas_energia_presentes(df)
    if not cols:
        print("No hay columnas principales de energía presentes.")
        return pd.DataFrame()
    print("\nTabla: energía global por año...")
    return df.groupby("year")[cols].sum(min_count=1).reset_index()
def tabla_co2_energy_global_anual(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["co2", "co2_per_capita", "primary_energy_consumption"]
    cols = [c for c in cols if c in df.columns]
    print("\nTabla: CO₂ + energía global por año...")
    return df.groupby("year")[cols].sum(min_count=1).reset_index()
def tabla_pais_anual(df: pd.DataFrame) -> pd.DataFrame:
    """
    CO2 + energía por pais - año
    Solo incluye paises con iso_code valido
    """
    print("\nTabla: país-año (CO₂ + energía)...")
    # Validación
    required = {"country", "year", "iso_code"}
    if not required.issubset(df.columns):
        faltan = required - set(df.columns)
        print(f"Faltan columnas requeridas: {faltan}")
        return pd.DataFrame
    metric_cols = [
        "co2",
        "co2_per_capita",
        "primary_energy_consumption",
        "carbon_intensity_raw",
    ] + columnas_energia_presentes(df)
    cols_base = ["country", "iso_code", "year"]
    cols = [c for c in cols_base + metric_cols if c in df.columns]
    
    # Filtrar solo paises con son iso_code
    out = df.loc[
        df["iso_code"].notna() &
        (df["iso_code"].astype(str).str.len() == 3),
        cols
    ].copy()
    
    return out
def tabla_decada_global(df: pd.DataFrame) -> pd.DataFrame:
    print("\n Tabla: CO₂ + energía global por década...")
    df_dec = df.copy()
    df_dec["decade"] = (df_dec["year"] // 10) * 10
    metric_cols = [
        "co2",
        "co2_per_capita",
        "primary_energy_consumption",
        "carbon_intensity_raw",
    ] + columnas_energia_presentes(df)
    metric_cols = [c for c in metric_cols if c in df_dec.columns]
    tabla = df_dec.groupby("decade")[metric_cols].agg(["sum", "mean"]).sort_index().reset_index()
    tabla.columns = ["_".join([str(c) for c in col if c]) for col in tabla.columns.values]
    return tabla
# ---------------------------------------------------------------------
# GRÁFICAS PRELIMINARES
# ---------------------------------------------------------------------
def plot_co2_vs_energy_world(df: pd.DataFrame, output_dir: Path):
    print("\nGráfico: CO₂ vs energía primaria - Mundo")
    df_world = df[df["country"] == "World"].sort_values("year")
    if df_world.empty:
        print("No se encontró 'World' en country.")
        return
    if "co2" not in df_world.columns or "primary_energy_consumption" not in df_world.columns:
        print("Faltan columnas 'co2' o 'primary_energy_consumption'.")
        return
    plt.figure()
    plt.plot(df_world["year"], df_world["co2"], label="CO₂")
    plt.plot(df_world["year"], df_world["primary_energy_consumption"], label="Energía primaria")
    plt.xlabel("Año")
    plt.ylabel("Nivel (sin ajustar unidades)")
    plt.title("CO₂ vs Energía primaria - Mundo")
    plt.legend()
    plt.tight_layout()
    out_path = output_dir / "co2_vs_energy_world.png"
    plt.savefig(out_path)
    plt.close()
    print(f"    {out_path.name}")
def plot_energy_mix_world(df: pd.DataFrame, output_dir: Path):
    print("\nGráfico: Consumo de energías en el Mundo")
    df_world = df[df["country"] == "World"].sort_values("year")
    cols = columnas_energia_presentes(df_world)
    if df_world.empty or not cols:
        print("No hay datos suficientes para el mix global.")
        return
    plt.figure()
    for col in cols:
        plt.plot(df_world["year"], df_world[col], label=col)
    plt.xlabel("Año")
    plt.ylabel("Consumo de energía (unidades TWh)")
    plt.title("Mix de energía mundial")
    plt.legend()
    plt.tight_layout()
    out_path = output_dir / "mix de energía mundial.png"
    plt.savefig(out_path)
    plt.close()
    print(f"    {out_path.name}")
def plot_scatter_co2_vs_energy_per_capita(df: pd.DataFrame, year: int, output_dir: Path):
    print(f"\n Gráfico: CO₂ per cápita vs energía per cápita ({year})")
    cols_needed = ["co2_per_capita", "energy_per_capita"]
    if not all(c in df.columns for c in cols_needed):
        print("Faltan columnas 'co2_per_capita' o 'energy_per_capita'.")
        return
    df_year = df[df["year"] == year].dropna(subset=cols_needed)
    if df_year.empty:
        print("No hay datos para el año indicado.")
        return
    plt.figure()
    plt.scatter(df_year["energy_per_capita"], df_year["co2_per_capita"])
    plt.xlabel("Energía per cápita")
    plt.ylabel("CO₂ per cápita")
    plt.title(f"CO₂ per cápita vs Energía per cápita - {year}")
    plt.tight_layout()
    out_path = output_dir / f"scatter_co2_vs_energy_per_capita_{year}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"    {out_path.name}")

def seleccionar_columnas_finales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selecciona únicamente las columnas finales requeridas:
    - llaves / ids (country, year, iso_code si existe)
    - CO2 mínimas
    - ENERGY_MAIN_COLS + ENERGY_CALC_COLS
    - derivadas propias (carbon_intensity_raw si existe)
    - renovables (installed_capacity_gw si existe)
    """
    final_cols: list[str] = []
    # llaves / ids
    for c in ["country", "year", "iso_code"]:
        if c in df.columns:
            final_cols.append(c)
    # CO2 mínimas
    final_cols += [c for c in CO2_MIN_COLS if c in df.columns]
    # energía main + calculadas
    final_cols += [c for c in (ENERGY_MAIN_COLS + ENERGY_CALC_COLS) if c in df.columns]
    # derivadas propias
    for c in ["carbon_intensity_raw"]:
        if c in df.columns:
            final_cols.append(c)
    # renovables
    for c in ["installed_capacity_gw"]:
        if c in df.columns:
            final_cols.append(c)
    # dedupe manteniendo orden
    final_cols = list(dict.fromkeys(final_cols))
    return df.loc[:, final_cols].copy()

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    # 1) Cargar datasets
    df_co2, df_energy = cargar_co2_energy(RUTA_CO2, RUTA_ENERGY)
    # 2) Limpiar PRIMERO CO2 (sin contar llaves)
    cols_no_contar_co2 = ["country", "year"]
    if "iso_code" in df_co2.columns:
        cols_no_contar_co2.append("iso_code")
    df_co2_clean = limpiar_filas_incompletas(
        df_co2,
        umbral=0.6,
        cols_no_contar=cols_no_contar_co2,
        tratar_ceros_como_na=True
    )
    # 3) Merge
    df_comb = mezclar_datasets(df_co2_clean, df_energy)
    # 3B) Capacidad instalada desde long_clean
    df_capacidad = construir_capacidad_desde_long_clean(RUTA_RENEW_LONG)
    # Agregar capacidad al combinado
    df_comb = agregar_capacidad_instalada(df_comb, df_capacidad)
    # 4) Limpieza POST-MERGE para quitar filas >60% en blanco del combinado
    if LIMPIAR_POST_MERGE:
        cols_no_contar_merge = ["country", "year"]
        if "iso_code" in df_comb.columns:
            cols_no_contar_merge.append("iso_code")
        df_clean = limpiar_filas_incompletas(
            df_comb,
            umbral=0.6,
            cols_no_contar=cols_no_contar_merge,
            tratar_ceros_como_na=True
        )
    else:
        df_clean = df_comb.copy()
    # 5) Intensidad de carbono
    df_clean = calcular_intensidad_carbono(df_clean)
    # 5.1) CO2 del sistema eléctrico (estimado)
    df_clean = calcular_co2_electrico_estimado(df_clean)
    # 5.2) Dejar solo columnas finales (ENERGY_MAIN_COLS + calculadas + CO2 mínimas + renovables)
    df_clean = seleccionar_columnas_finales(df_clean)
    print(f"   ➜ Dataset final (columnas filtradas): {df_clean.shape}")
    # 6) Guardar dataset combinado limpio
    out_csv = OUTPUT_DIR / "co2_energy_combined_clean.csv"
    df_clean.to_csv(out_csv, index=False)
    print(f"\n Dataset combinado limpio guardado en: {out_csv.resolve()}")
    # 7) EDA básico
    eda_basico(df_clean, nombre="co2_energy_combined_clean")
    eda_completo_dataset_combinado(df_clean, output_dir=OUTPUT_DIR, nombre="co2_energy_combined_clean")
    # 8) Tablas agregadas
    tabla_energia_global = tabla_energia_global_anual(df_clean)
    if not tabla_energia_global.empty:
        tabla_energia_global.to_csv(OUTPUT_DIR / "tabla_energia_global_anual.csv", index=False)
        print("    tabla_energia_global_anual.csv guardada")
    tabla_co2_energy_global = tabla_co2_energy_global_anual(df_clean)
    tabla_co2_energy_global.to_csv(OUTPUT_DIR / "tabla_co2_energy_global_anual.csv", index=False)
    print("    tabla_co2_energy_global_anual.csv guardada")
    tabla_pais = tabla_pais_anual(df_clean)
    tabla_pais.to_csv(OUTPUT_DIR / "tabla_pais_anual.csv", index=False)
    print("    tabla_pais_anual.csv guardada")
    tabla_decada = tabla_decada_global(df_clean)
    tabla_decada.to_csv(OUTPUT_DIR / "tabla_decada_global.csv", index=False)
    print("    tabla_decada_global.csv guardada")
    # 9) Gráficas preliminares
    plot_co2_vs_energy_world(df_clean, OUTPUT_DIR)
    plot_energy_mix_world(df_clean, OUTPUT_DIR)
    plot_scatter_co2_vs_energy_per_capita(df_clean, year=2023, output_dir=OUTPUT_DIR)
    print("\n Proceso combinado CO₂ + Energía completado.")
if __name__ == "__main__":
    main()