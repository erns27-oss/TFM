import pandas as pd

# Cargar resultados del clustering
df = pd.read_csv("outputs/clusters_energy_countries_iso3.csv")

FEATURES = [
    "energy_per_capita",
    "carbon_intensity_raw",
    "electricity_share_energy",
    "renewables_share",
    "co2_per_capita",
    "energy_per_gdp",
]


# -----------------------------
# Listas de países por región
# -----------------------------
europe = [
    "AUT","BEL","BGR","CHE","CYP","CZE","DEU","DNK","ESP","EST","FIN","FRA",
    "GBR","GRC","HRV","HUN","IRL","ITA","LTU","LUX","LVA","MLT","NLD","NOR",
    "POL","PRT","ROU","SVK","SVN","SWE"
]

south_america = [
    "ARG","BOL","BRA","CHL","COL","ECU","GUY","PER","PRY","SUR","URY","VEN"
]

# Filtrar
eu_df = df[df["iso_code"].isin(europe)]
sa_df = df[df["iso_code"].isin(south_america)]

# -----------------------------
# Distribución de clusters
# -----------------------------
eu_clusters = eu_df["cluster"].value_counts().sort_index()
sa_clusters = sa_df["cluster"].value_counts().sort_index()

print("Europa – distribución de clusters")
print(eu_clusters)

print("\nSouth America – distribución de clusters")
print(sa_clusters)

# Resumen por cluster
summary = (
    df.groupby("cluster")[FEATURES]
      .mean()
      .round(3)
)

# Número de países por cluster
counts = df.groupby("cluster").size().rename("n_countries")

summary_table = summary.join(counts)

print(summary_table)

# Guardar para Power BI / TFM
summary_table.to_csv("outputs/cluster_summary_table.csv")
