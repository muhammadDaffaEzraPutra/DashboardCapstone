import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Judul dashboard
st.title("Visualisasi Cluster Kondisi Jalan Kalimantan Timur")

# Load data jalan
df = pd.read_csv("dashboard.csv")

# Load GeoJSON kabupaten/kota Kalimantan Timur
with open("Kaltim.geojson") as f:
    geojson = json.load(f)

# Sidebar filter berdasarkan kabupaten/kota
daftar_kabupaten = df["Kec._Yang_dilalui"].unique()
pilihan_kecamatan = st.sidebar.multiselect(
    "Pilih Kecamatan yang Dilalui", daftar_kabupaten, default=daftar_kabupaten
)

# Filter data
filtered_df = df[df["Kec._Yang_dilalui"].isin(pilihan_kecamatan)].copy()

# Lakukan clustering pada data yang difilter
features = filtered_df[['Panjang_Ruas_(Km)', 'Total_Kerusakan']]
kmeans = KMeans(n_clusters=3, random_state=42)
filtered_df['cluster'] = kmeans.fit_predict(features)


# Tampilkan peta cluster per kabupaten
fig = px.choropleth_mapbox(

    filtered_df,
    geojson=geojson,
    locations="Kec._Yang_dilalui",               # Sesuaikan dengan kolom yang cocok dengan geojson NAME_2
    featureidkey="properties.NAME_2",         # Kolom nama kabupaten/kota di geojson
    color="cluster",
    mapbox_style="carto-positron",
    zoom=5,
    center={"lat": -0.5, "lon": 117.5},
    opacity=0.6,
    hover_name="Kec._Yang_dilalui",
    hover_data=["Nama_Ruas_Jalan", "Panjang_Ruas_(Km)", "Kategori_Kerusakan"],
)

st.plotly_chart(fig, use_container_width=True)

# Scatter Plot Panjang vs Kerusakan
st.subheader("Scatter Plot: Panjang Ruas vs Total Kerusakan")
fig2, ax = plt.subplots()
sns.scatterplot(data=filtered_df, x='Panjang_Ruas_(Km)', y='Total_Kerusakan', hue='cluster', palette='Set1', ax=ax)
st.pyplot(fig2)

# Boxplot Total Kerusakan per Cluster
st.subheader("Distribusi Total Kerusakan per Cluster")
fig3, ax = plt.subplots()
sns.boxplot(data=filtered_df, x='cluster', y='Total_Kerusakan', palette='Set2', ax=ax)
st.pyplot(fig3)

# Boxplot Panjang Ruas per Cluster
st.subheader("Distribusi Panjang Ruas per Cluster")
fig4, ax = plt.subplots()
sns.boxplot(data=filtered_df, x='cluster', y='Panjang_Ruas_(Km)', palette='Set3', ax=ax)
st.pyplot(fig4)

# Bar Chart Jumlah Ruas Jalan per Cluster
st.subheader("Jumlah Ruas Jalan per Cluster")
cluster_counts = filtered_df['cluster'].value_counts().sort_index()
st.bar_chart(cluster_counts)

# Heatmap Korelasi Fitur
st.subheader("Heatmap Korelasi Fitur")
numeric_cols = filtered_df.select_dtypes(include=np.number)
correlation = numeric_cols.corr()
fig5, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig5)
