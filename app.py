# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import requests
import plotly.graph_objects as go

from modelo_emerrel import ejecutar_modelo
from meteobahia import preparar_para_modelo, usar_fechas_de_input, reiniciar_feb_oct
from meteobahia_api import fetch_meteobahia_api_xml

# ===================== CONFIG =====================
st.set_page_config(page_title="🌾 EMERGENCIA AGRÍCOLA", layout="wide")
st.title("🌾 Predicción de Emergencia Agrícola - HIRSHIN")

API_URL = "https://meteobahia.com.ar/scripts/forecast/for-bd.xml"
HISTORICO_URL = "https://raw.githubusercontent.com/PREDWEEM/HIRSHIN2/main/data/historico.xlsx"

EMEAC_MIN = 5
EMEAC_MAX = 7
EMEAC_DEF = 6
FORZAR_UMBRAL = False

# ===================== SIDEBAR =====================
st.sidebar.header("Configuración")
usar_codigo = st.sidebar.checkbox("Forzar umbral desde código", value=FORZAR_UMBRAL)
umbral_slider = st.sidebar.slider("Umbral ajustable", EMEAC_MIN, EMEAC_MAX, EMEAC_DEF)
umbral = EMEAC_DEF if usar_codigo else umbral_slider

# API config
st.sidebar.subheader("API Pronóstico")
token = st.sidebar.text_input("Token Bearer (opcional)", type="password")
usar_headers = st.sidebar.checkbox("Compatibilidad navegador", value=True)
if st.sidebar.button("🔄 Recargar datos"):
    st.cache_data.clear()

# ===================== CARGAR DATOS =====================
@st.cache_data(ttl=600)
def cargar_api(url, token, usar_headers):
    return fetch_meteobahia_api_xml(url, token=token, use_browser_headers=usar_headers)

@st.cache_data(ttl=600)
def cargar_historico(url):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return pd.read_excel(io.BytesIO(r.content))

# API
with st.spinner("Descargando pronóstico..."):
    df_api = cargar_api(API_URL, token, usar_headers)

# Limitar a los primeros 7 días
df_api["Fecha"] = pd.to_datetime(df_api["Fecha"])
df_api = df_api.sort_values("Fecha")
dias_unicos = df_api["Fecha"].dt.normalize().unique()
df_api = df_api[df_api["Fecha"].dt.normalize().isin(dias_unicos[:7])]

if df_api.empty:
    st.error("No se pudieron obtener datos del pronóstico.")
    st.stop()

# Histórico
with st.spinner("Descargando histórico..."):
    df_hist = cargar_historico(HISTORICO_URL)

# ===================== NORMALIZAR HISTÓRICO =====================
def normalizar_hist(df, year):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    renombrar = {
        "fecha": "Fecha", "tmax": "TMAX", "tmin": "TMIN", "prec": "Prec", "julian_days": "Julian_days"
    }
    for k, v in renombrar.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)

    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"])
    if "Julian_days" not in df.columns and "Fecha" in df.columns:
        df["Julian_days"] = df["Fecha"].dt.dayofyear
    if "Fecha" not in df.columns and "Julian_days" in df.columns:
        df["Fecha"] = pd.Timestamp(f"{year}-01-01") + pd.to_timedelta(df["Julian_days"] - 1, unit="D")

    for col in ["TMAX", "TMIN", "Prec"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"])
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    return df[["Fecha", "Julian_days", "TMAX", "TMIN", "Prec"]]

año_api = df_api["Fecha"].min().year
df_hist = normalizar_hist(df_hist, año_api)
df_hist = df_hist[df_hist["Fecha"] < df_api["Fecha"].min()]

# Fusionar histórico + API
df_all = pd.concat([df_hist, df_api], ignore_index=True).drop_duplicates("Fecha")
df_all = df_all.sort_values("Fecha")
df_all["Julian_days"] = df_all["Fecha"].dt.dayofyear

# ===================== PREPARAR Y EJECUTAR MODELO =====================
input_df = preparar_para_modelo(df_all)
if input_df.empty:
    st.error("No se pudieron preparar los datos para el modelo.")
    st.stop()

# Cargar pesos
try:
    IW = np.load("IW.npy")
    bias_IW = np.load("bias_IW.npy")
    LW = np.load("LW.npy")
    bias_out = np.load("bias_out.npy")
except Exception as e:
    st.error(f"Error cargando pesos del modelo: {e}")
    st.stop()

# Ejecutar modelo
resultado = ejecutar_modelo(input_df, IW, bias_IW, LW, bias_out, umbral)
fechas_originales = usar_fechas_de_input(df_all, len(resultado))
if fechas_originales is not None:
    resultado["Fecha"] = fechas_originales

# ===================== PROCESAR SALIDA =====================
resultado_rango = reiniciar_feb_oct(resultado[["Fecha", "EMERREL (0-1)"]].copy(), umbral_ajustable=umbral)
if resultado_rango.empty:
    st.warning("No hay datos entre 1-feb y 1-oct.")
    st.stop()

resultado_rango["MA5"] = resultado_rango["EMERREL (0-1)"].rolling(5, min_periods=1).mean()
resultado_rango["Nivel"] = resultado_rango["EMERREL (0-1)"].apply(
    lambda x: "Bajo" if x < 0.2 else "Medio" if x < 0.4 else "Alto"
)

# Cálculo acumulado
emeac_acumulado = resultado_rango["EMERREL (0-1)"].cumsum()
emeac_pct = np.clip(emeac_acumulado / umbral * 100, 0, 100)

# ===================== GRÁFICO EMERREL =====================
st.subheader("📊 Emergencia Relativa Diaria (EMERREL)")
fig1 = go.Figure()
colores = {"Bajo": "green", "Medio": "orange", "Alto": "red"}
fig1.add_bar(
    x=resultado_rango["Fecha"],
    y=resultado_rango["EMERREL (0-1)"],
    marker_color=resultado_rango["Nivel"].map(colores),
    name="EMERREL"
)
fig1.add_trace(go.Scatter(x=resultado_rango["Fecha"], y=resultado_rango["MA5"], name="Media móvil 5 días", mode="lines"))
fig1.update_layout(
    yaxis_title="EMERREL (0-1)",
    hovermode="x unified",
    height=500
)
st.plotly_chart(fig1, use_container_width=True)

# ===================== GRÁFICO EMEAC % =====================
st.subheader("📈 Emergencia Acumulada (EMEAC %)")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=resultado_rango["Fecha"], y=emeac_pct, name="EMEAC (%)", mode="lines", line=dict(width=3)))
for y in [25, 50, 75, 90]:
    fig2.add_hline(y=y, line_dash="dash", annotation_text=f"{y}%", opacity=0.3)
fig2.update_layout(
    yaxis_title="EMEAC (%)",
    hovermode="x unified",
    height=500,
    yaxis_range=[0, 100]
)
st.plotly_chart(fig2, use_container_width=True)

# ===================== TABLA DE RESULTADOS =====================
st.subheader("📋 Tabla de Resultados (con colores)")
emoji = {"Bajo": "🟢 Bajo", "Medio": "🟡 Medio", "Alto": "🔴 Alto"}

tabla_vis = pd.DataFrame({
    "Fecha": resultado_rango["Fecha"],
    "Día juliano": resultado_rango["Fecha"].dt.dayofyear,
    "Nivel EMERREL": resultado_rango["Nivel"].map(emoji),
    "EMEAC (%)": emeac_pct.round(1)
})

st.dataframe(tabla_vis, use_container_width=True)

# Exportar CSV sin emojis
tabla_csv = pd.DataFrame({
    "Fecha": resultado_rango["Fecha"],
    "Día juliano": resultado_rango["Fecha"].dt.dayofyear,
    "Nivel EMERREL": resultado_rango["Nivel"],
    "EMEAC (%)": emeac_pct.round(1)
})
csv = tabla_csv.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar CSV", data=csv, file_name="resultados_emeac.csv", mime="text/csv")

