from pathlib import Path

# Contenido completo del archivo app.py con las modificaciones
modified_app_py = """
# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import io
import requests

from modelo_emerrel import ejecutar_modelo
from meteobahia import (
    preparar_para_modelo,
    usar_fechas_de_input,
    reiniciar_feb_oct,
)
from meteobahia_api import fetch_meteobahia_api_xml  # usa headers tipo navegador

# ==== Plotly (opcional) ====
try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

st.set_page_config(page_title="PREDICCION EMERGENCIA AGRICOLA HIRSHIN", layout="wide")

# ====================== UMBRALES EMEAC (EDITABLES EN CÓDIGO) ======================
EMEAC_MIN = 5
EMEAC_MAX = 7
EMEAC_MIN, EMEAC_MAX = sorted([EMEAC_MIN, EMEAC_MAX])
EMEAC_AJUSTABLE_DEF = 6
FORZAR_AJUSTABLE_DESDE_CODIGO = False

# ====================== Config fija (no visible) ======================
DEFAULT_API_URL  = "https://meteobahia.com.ar/scripts/forecast/for-bd.xml"
DEFAULT_HIST_URL = "https://raw.githubusercontent.com/PREDWEEM/HIRSHIN2/main/data/historico.xlsx"

# ====================== Estado persistente ======================
if "api_token" not in st.session_state:
    st.session_state["api_token"] = ""
if "reload_nonce" not in st.session_state:
    st.session_state["reload_nonce"] = 0
if "compat_headers" not in st.session_state:
    st.session_state["compat_headers"] = True

# ================= Sidebar =================
st.sidebar.header("Fuente de datos")
fuente = st.sidebar.radio(
    "Elegí cómo cargar datos",
    options=["API + Histórico", "Subir Excel"],
    index=0,
)

usar_codigo = st.sidebar.checkbox(
    "Usar umbral ajustable desde CÓDIGO",
    value=FORZAR_AJUSTABLE_DESDE_CODIGO
)

umbral_slider = st.sidebar.slider(
    "Seleccione el umbral EMEAC (Ajustable)",
    min_value=int(EMEAC_MIN),
    max_value=int(EMEAC_MAX),
    value=int(np.clip(EMEAC_AJUSTABLE_DEF, EMEAC_MIN, EMEAC_MAX))
)

umbral_usuario = int(np.clip(
    EMEAC_AJUSTABLE_DEF if usar_codigo else umbral_slider,
    EMEAC_MIN, EMEAC_MAX
))

# Token y headers
st.sidebar.text_input("Bearer token (opcional)", key="api_token", type="password")
st.session_state["compat_headers"] = st.sidebar.checkbox(
    "Compatibilidad (headers de navegador)", value=st.session_state["compat_headers"]
)

# ✅ Nuevo checkbox
usar_solo_7_dias = st.sidebar.checkbox(
    "Usar solo 7 días de pronóstico (mejor precisión)", value=True
)

if st.sidebar.button("Actualizar ahora (forzar recarga)"):
    st.session_state["reload_nonce"] += 1

# ================= Helpers =================
@st.cache_data(ttl=600)
def fetch_api_cached(url: str, token: str | None, nonce: int, use_browser_headers: bool):
    return fetch_meteobahia_api_xml(url.strip(), token=token or None, use_browser_headers=use_browser_headers)

def read_hist_from_url(url: str) -> pd.DataFrame:
    if not url.strip():
        return pd.DataFrame()
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url.strip(), headers=headers, timeout=25)
        r.raise_for_status()
        buf = io.BytesIO(r.content)
        if url.lower().endswith(".csv"):
            return pd.read_csv(buf)
        return pd.read_excel(buf)
    except Exception as e:
        st.error(f"No pude descargar el histórico desde la URL: {e}")
        return pd.DataFrame()

# ================= Flujo principal =================
st.title("PREDICCION EMERGENCIA AGRICOLA HIRSHIN")

input_df_raw = None
source_label = None

if fuente == "API + Histórico":
    api_url = DEFAULT_API_URL
    token = st.session_state["api_token"] or ""
    compat = bool(st.session_state["compat_headers"])

    # 1) API
    df_api = pd.DataFrame()
    if api_url.strip():
        try:
            with st.spinner("Descargando API…"):
                df_api = fetch_api_cached(api_url, token, st.session_state["reload_nonce"], compat)

            if df_api.empty:
                st.warning("API: sin filas.")
            else:
                # ✅ Filtro de los primeros 7 días si el checkbox está activado
                if usar_solo_7_dias:
                    min_fecha_api = pd.to_datetime(df_api["Fecha"].min())
                    max_fecha_7 = min_fecha_api + pd.Timedelta(days=6)
                    df_api = df_api[df_api["Fecha"] <= max_fecha_7]

        except Exception as e:
            st.error(f"Error API: {e}")

    # (resto del script continúa normalmente…)
"""

# Guardar archivo
output_path = Path("/mnt/data/app_modificado.py")
output_path.write_text(modified_app_py)

output_path.name  # para mostrar el nombre del archivo generado

