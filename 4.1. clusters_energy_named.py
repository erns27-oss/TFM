import pandas as pd

df = pd.read_csv("outputs/clusters_energy_countries_iso3.csv")

cluster_names = {
    0: "Economías en transición energética",
    1: "Renovables estructurales y baja intensidad",
    2: "Alta intensidad energética y sistemas maduros"
}

df["cluster_name"] = df["cluster"].map(cluster_names)

df.to_csv("outputs/clusters_energy_countries_iso3_named.csv", index=False)