# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import io
import requests
import time

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

# Umbrales EMERREL (riesgo diario)
THR_BAJO_MEDIO = 0.020
THR_MEDIO_ALTO = 0.079
NIVEL_ICONO = {"Bajo": "🟢 Bajo", "Medio": "🟠 Medio", "Alto": "🔴 Alto"}

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

# ====================== Utilidades de estabilidad / UX ======================
def clear_and_rerun():
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
    except Exception:
        pass
    st.rerun()

def sidebar_refresh_button(label: str = "🔄 Refrescar datos"):
    with st.sidebar:
        if st.button(label, key="btn_refresh_sidebar"):
            clear_and_rerun()

def with_retries(fn, retries: int = 2, delay: float = 0.8):
    def _wrapped(*args, **kwargs):
        last_err = None
        for i in range(retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_err = e
                if i < retries:
                    time.sleep(delay)
        raise last_err
    return _wrapped

# ============== Sidebar ==============
st.sidebar.header("Fuente de datos")
fuente = st.sidebar.radio(
    "Elegí cómo cargar datos",
    options=["API + Histórico", "Subir Excel"],
    index=0,
    key="radio_fuente"
)

usar_codigo = st.sidebar.checkbox(
    "Usar umbral ajustable desde CÓDIGO",
    value=FORZAR_AJUSTABLE_DESDE_CODIGO,
    key="chk_usar_codigo"
)

umbral_slider = st.sidebar.slider(
    "Seleccione el umbral EMEAC (Ajustable)",
    min_value=int(EMEAC_MIN),
    max_value=int(EMEAC_MAX),
    value=int(np.clip(EMEAC_AJUSTABLE_DEF, EMEAC_MIN, EMEAC_MAX)),
    key="sld_umbral"
)

umbral_usuario = int(np.clip(
    EMEAC_AJUSTABLE_DEF if usar_codigo else umbral_slider,
    EMEAC_MIN, EMEAC_MAX
))

# Opción de compatibilidad de headers + recarga
st.session_state["compat_headers"] = st.sidebar.checkbox(
    "Compatibilidad (headers de navegador)", value=st.session_state["compat_headers"], key="chk_headers"
)
if st.sidebar.button("Actualizar ahora (forzar recarga)", key="btn_reload_now"):
    st.session_state["reload_nonce"] += 1

sidebar_refresh_button()

# ============== Helpers ==============
@st.cache_data(ttl=600, show_spinner=False)
def fetch_api_cached(url: str, token: str | None, nonce: int, use_browser_headers: bool):
    fn = with_retries(
        lambda: fetch_meteobahia_api_xml(url.strip(), token=token or None, use_browser_headers=use_browser_headers),
        retries=2, delay=0.8
    )
    return fn()

@st.cache_data(ttl=3600, show_spinner=False)
def read_hist_from_url(url: str) -> pd.DataFrame:
    if not url.strip():
        return pd.DataFrame()
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url.strip(), headers=headers, timeout=25)
    r.raise_for_status()
    buf = io.BytesIO(r.content)
    if url.lower().endswith(".csv"):
        return pd.read_csv(buf)
    return pd.read_excel(buf)

def normalize_hist(df_hist: pd.DataFrame, api_year: int) -> pd.DataFrame:
    import calendar
    df = df_hist.copy()
    df.columns = [str(c).strip() for c in df.columns]
    low2orig = {c.lower(): c for c in df.columns}
    def has(c): return c in low2orig
    def col(c): return low2orig[c]

    ren = {}
    for cands, tgt in [
        (["fecha", "date"], "Fecha"),
        (["julian_days", "julian"], "Julian_days"),
        (["tmax"], "TMAX"),
        (["tmin"], "TMIN"),
        (["prec", "ppt", "lluvia"], "Prec"),
    ]:
        for c in cands:
            if has(c):
                ren[col(c)] = tgt
                break
    df = df.rename(columns=ren)

    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    for c in ["TMAX", "TMIN", "Prec", "Julian_days"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    leap = calendar.isleap(int(api_year))
    max_j = 366 if leap else 365
    if "Julian_days" in df.columns:
        df = df[df["Julian_days"].between(1, max_j)]

    if "Fecha" not in df.columns and "Julian_days" in df.columns:
        base = pd.Timestamp(int(api_year), 1, 1)
        df["Fecha"] = df["Julian_days"].astype(int).apply(lambda d: base + pd.Timedelta(days=d - 1))
    if "Julian_days" not in df.columns and "Fecha" in df.columns:
        df["Julian_days"] = df["Fecha"].dt.dayofyear

    req = {"Fecha", "Julian_days", "TMAX", "TMIN", "Prec"}
    faltan = req - set(df.columns)
    if faltan:
        raise ValueError(f"Histórico sin columnas requeridas: {faltan}")

    df = df.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    return df[["Fecha", "Julian_days", "TMAX", "TMIN", "Prec"]]

# ================= Flujo principal =================
st.title("PREDICCION EMERGENCIA AGRICOLA HIRSHIN")

input_df_raw = None
source_label = None

if fuente == "API + Histórico":
    api_url = DEFAULT_API_URL

    # Token sin label visible (opcional)
    st.sidebar.text_input(label=" ", key="api_token", type="password", label_visibility="collapsed")

    token = st.session_state["api_token"] or ""
    compat = bool(st.session_state["compat_headers"])

    with st.spinner("Descargando pronóstico..."):
        try:
            df_api = fetch_api_cached(api_url, token, st.session_state["reload_nonce"], compat)
        except Exception as e:
            st.error(f"No se pudieron obtener datos del pronóstico: {e}")
            st.stop()

    # Asegurar fechas + recortar a primeros 8 días únicos
    df_api["Fecha"] = pd.to_datetime(df_api["Fecha"], errors="coerce")
    df_api = df_api.dropna(subset=["Fecha"]).sort_values("Fecha")
    dias_unicos = df_api["Fecha"].dt.normalize().unique()
    df_api = df_api[df_api["Fecha"].dt.normalize().isin(dias_unicos[:8])]

    if df_api.empty:
        st.error("No se pudieron obtener datos del pronóstico.")
        st.stop()

    with st.spinner("Descargando histórico..."):
        try:
            dfh_raw = read_hist_from_url(DEFAULT_HIST_URL)
        except Exception as e:
            st.error(f"No pude descargar el histórico: {e}")
            dfh_raw = pd.DataFrame()

    min_api_date = pd.to_datetime(df_api["Fecha"].min()).normalize()
    api_year = int(min_api_date.year)
    start_hist = pd.Timestamp(api_year, 1, 1)
    end_hist = min_api_date - pd.Timedelta(days=1)

    df_hist_trim = pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    if not dfh_raw.empty and end_hist >= start_hist:
        try:
            df_hist_all = normalize_hist(dfh_raw, api_year=api_year)
            df_hist_trim = df_hist_all[(df_hist_all["Fecha"]>=start_hist)&(df_hist_all["Fecha"]<=end_hist)]
        except Exception as e:
            st.error(f"Error normalizando histórico: {e}")

    df_all = pd.concat([df_hist_trim, df_api], ignore_index=True)
    df_all["Fecha"] = pd.to_datetime(df_all["Fecha"], errors="coerce")
    df_all = df_all.dropna(subset=["Fecha"]).sort_values("Fecha").drop_duplicates(subset=["Fecha"])
    df_all["Julian_days"] = df_all["Fecha"].dt.dayofyear

    if df_all.empty:
        st.error("Fusión vacía.")
        st.stop()

    input_df_raw = df_all.copy()
    src = ["API"]
    if not df_hist_trim.empty:
        src.append(f"Hist ({df_hist_trim['Fecha'].min().date()} → {df_hist_trim['Fecha'].max().date()})")
    source_label = " + ".join(src)

elif fuente == "Subir Excel":
    uploaded_file = st.file_uploader("Cargar archivo input.xlsx", type=["xlsx"], key="upl_excel_input")
    if uploaded_file is not None:
        try:
            input_df_raw = pd.read_excel(uploaded_file)
            source_label = f"Excel: {uploaded_file.name}"
        except Exception as e:
            st.error(f"No pude leer el Excel: {e}")

# ================= Preparación para el modelo =================
if input_df_raw is None or input_df_raw.empty:
    st.stop()

# Normalizar columnas
base_cols = ["Fecha", "Julian_days", "TMAX", "TMIN", "Prec"]
for c in base_cols:
    if c not in input_df_raw.columns:
        st.error(f"Falta la columna requerida: {c}")
        st.stop()

df_in = input_df_raw.copy()
df_in = df_in.sort_values("Fecha").reset_index(drop=True)

# Preprocesamiento específico del pipeline (si aplica)
try:
    df_model = preparar_para_modelo(df_in.copy())
    if df_model is None or df_model.empty:
        df_model = df_in.copy()
except Exception:
    df_model = df_in.copy()

# ================= Ejecutar modelo =================
pred = None
err_modelo = None

# Intentos con distintas firmas para máxima compatibilidad
try:
    pred = ejecutar_modelo(df_model, emeac_den=umbral_usuario)
except TypeError:
    try:
        pred = ejecutar_modelo(df_model, umbral=umbral_usuario)
    except TypeError:
        try:
            pred = ejecutar_modelo(df_model)
        except Exception as e:
            err_modelo = str(e)
    except Exception as e:
        err_modelo = str(e)
except Exception as e:
    err_modelo = str(e)

if pred is None:
    st.error(f"No se pudo ejecutar el modelo. Detalle: {err_modelo or 'desconocido'}")
    st.stop()

# Asegurar merge por Fecha
if "Fecha" not in pred.columns:
    # Si el modelo devuelve por índice, lo reconstruimos
    pred = pred.reset_index().rename(columns={"index": "Fecha"})
    try:
        pred["Fecha"] = pd.to_datetime(pred["Fecha"], errors="coerce")
    except Exception:
        pred["Fecha"] = df_in["Fecha"].values

df_pred = pd.merge(df_in, pred, on="Fecha", how="left")

# Nombres esperados/flexibles
# EMERREL diario -> buscar varias variantes
emerrel_candidates = [c for c in df_pred.columns if str(c).lower() in ("emerrel","emergencia_relativa","em_rel","em_rel_diaria")]
emeac_candidates   = [c for c in df_pred.columns if str(c).lower() in ("emeac","emergencia_acumulada","em_ac","em_acum","emerac")]

if emerrel_candidates:
    col_emerrel = emerrel_candidates[0]
else:
    # fallback conservador si el modelo no lo provee
    col_emerrel = "EMERREL"
    df_pred[col_emerrel] = np.nan

if emeac_candidates:
    col_emeac = emeac_candidates[0]
else:
    col_emeac = "EMEAC"
    # construir acumulada aproximada si no vino (integrando EMERREL)
    if df_pred[col_emerrel].notna().any():
        df_pred[col_emeac] = df_pred[col_emerrel].fillna(0).cumsum()
    else:
        df_pred[col_emeac] = np.nan

# ================= Clasificación de riesgo =================
def nivel_emerg_rel(x):
    if pd.isna(x):
        return np.nan
    if x < THR_BAJO_MEDIO:
        return "Bajo"
    if x < THR_MEDIO_ALTO:
        return "Medio"
    return "Alto"

df_pred["Nivel_EMERREL"] = df_pred[col_emerrel].apply(nivel_emerg_rel)

# Regla de seguridad: si EMEAC (%) < 10, forzar "Bajo" (si la métrica está en 0–1, interpreta 0.10)
try:
    df_pred["Nivel_EMERREL"] = np.where(
        (df_pred[col_emeac].notna()) & (df_pred[col_emeac] < 0.10),
        "Bajo",
        df_pred["Nivel_EMERREL"]
    )
except Exception:
    pass

df_pred["Nivel_icono"] = df_pred["Nivel_EMERREL"].map(NIVEL_ICONO).fillna("")

# ================= Vistas =================
st.caption(f"Fuente: {source_label} · Registros: {len(df_pred)} · Umbral EMEAC ajustable = **{umbral_usuario}**")

# -------- Gráfico 1: EMERREL (barras coloreadas) --------
with st.container():
    st.subheader("Emergencia relativa (EMERREL) diaria")
    fig1, ax1 = plt.subplots(figsize=(10, 3.6))
    fechas = df_pred["Fecha"]
    y = df_pred[col_emerrel].astype(float)

    # Colores por nivel
    colores = df_pred["Nivel_EMERREL"].map(
        {"Bajo": "#2ca02c", "Medio": "#ff7f0e", "Alto": "#d62728"}
    ).fillna("#808080")

    ax1.bar(fechas, y, width=0.85, align="center", edgecolor="none", color=colores)

    # Líneas de umbral (sin leyenda de “pronóstico”)
    ax1.axhline(THR_BAJO_MEDIO, linestyle="--", linewidth=1)
    ax1.axhline(THR_MEDIO_ALTO, linestyle="--", linewidth=1)

    ax1.set_ylabel("EMERREL")
    ax1.set_xlabel("")
    ax1.set_xlim(fechas.min(), fechas.max())
    ax1.grid(True, axis="y", alpha=0.25)

    # Leyenda propia (niveles)
    legend_patches = [
        Patch(color="#2ca02c", label="Bajo"),
        Patch(color="#ff7f0e", label="Medio"),
        Patch(color="#d62728", label="Alto"),
    ]
    ax1.legend(handles=legend_patches, loc="upper left")
    st.pyplot(fig1)
    plt.close(fig1)

# -------- Gráfico 2: EMEAC (línea) + umbral ajustable --------
with st.container():
    st.subheader("Emergencia acumulada (EMEAC)")
    fig2, ax2 = plt.subplots(figsize=(10, 3.6))
    ax2.plot(df_pred["Fecha"], df_pred[col_emeac].astype(float))
    ax2.axhline(umbral_usuario, linestyle=":", linewidth=1.2)  # línea de referencia del umbral ajustable
    ax2.set_ylabel("EMEAC")
    ax2.set_xlabel("")
    ax2.set_xlim(df_pred["Fecha"].min(), df_pred["Fecha"].max())
    ax2.grid(True, axis="y", alpha=0.25)
    st.pyplot(fig2)
    plt.close(fig2)

# -------- Tabla resumida --------
st.subheader("Tabla diaria")
cols_tabla = ["Fecha", "Julian_days", "TMAX", "TMIN", "Prec", col_emerrel, col_emeac, "Nivel_icono"]
tabla = df_pred.loc[:, [c for c in cols_tabla if c in df_pred.columns]].copy()
# Redondeos amables
for c in [col_emerrel, col_emeac, "TMAX", "TMIN", "Prec"]:
    if c in tabla.columns:
        tabla[c] = pd.to_numeric(tabla[c], errors="coerce").round(3)

st.dataframe(tabla, use_container_width=True)

# -------- Descarga --------
csv = tabla.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar CSV", data=csv, file_name="prediccion_hirshin.csv", mime="text/csv")

# -------- Notas --------
with st.expander("Notas"):
    st.markdown(
        f"""
- **Pronóstico recortado a 8 días:** se toman únicamente los primeros 8 días únicos del feed XML.
- **Umbral EMEAC ajustable:** actualmente en **{umbral_usuario}** (rango permitido: {EMEAC_MIN}–{EMEAC_MAX}).
- **Regla de seguridad:** si EMEAC < 0.10 (escala 0–1), el nivel diario se fuerza a **Bajo**.
- Si tu módulo `ejecutar_modelo` devuelve otros nombres de columnas, el script intenta detectarlos (EMERREL/EMEAC) y se adapta.
        """
    )
