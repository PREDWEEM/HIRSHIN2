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

# ========= Page config =========
st.set_page_config(
    page_title="PREDICCION EMERGENCIA AGRICOLA HIRSHIN",
    layout="wide",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": None,
    },
)

# ====================== UMBRALES EMEAC ======================
EMEAC_MIN = 5
EMEAC_MAX = 7
EMEAC_MIN, EMEAC_MAX = sorted([EMEAC_MIN, EMEAC_MAX])
EMEAC_AJUSTABLE_DEF = 6
FORZAR_AJUSTABLE_DESDE_CODIGO = False  # True ignora el slider y usa EMEAC_AJUSTABLE_DEF

# ====================== Config fija (no visible) ======================
DEFAULT_API_URL  = "https://meteobahia.com.ar/scripts/forecast/for-bd.xml"  # no se expone en UI
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

# Forzar umbral desde código (checkbox sin texto visible)
usar_codigo = st.sidebar.checkbox(
    label=" ",
    value=FORZAR_AJUSTABLE_DESDE_CODIGO,
    key="chk_usar_codigo",
    label_visibility="collapsed"
)

# Slider de umbral (solo visible si no se fuerza desde código)
umbral_slider = st.sidebar.slider(
    "Seleccione el umbral EMEAC (Ajustable)",
    min_value=int(EMEAC_MIN),
    max_value=int(EMEAC_MAX),
    value=int(np.clip(EMEAC_AJUSTABLE_DEF, EMEAC_MIN, EMEAC_MAX)),
)

# Checkbox de compatibilidad de headers (sin leyenda visible)
st.session_state["compat_headers"] = st.sidebar.checkbox(
    label=" ",
    value=st.session_state["compat_headers"],
    label_visibility="collapsed"
)

# Umbral efectivo que usa la app
umbral_usuario = int(np.clip(
    EMEAC_AJUSTABLE_DESDE_CODIGO if usar_codigo else umbral_slider,
    EMEAC_MIN, EMEAC_MAX
)) if False else int(np.clip(  # fallback para evitar NameError si se renombra
    EMEAC_AJUSTABLE_DEF if usar_codigo else umbral_slider,
    EMEAC_MIN, EMEAC_MAX
))

# ============== Helpers =================
@st.cache_data(ttl=600)
def fetch_api_cached(url: str, token: str | None, nonce: int, use_browser_headers: bool):
    """
    Wrapper con caché para la API. 'nonce' invalida la caché cuando se aprieta 'Actualizar'.
    Retorna DataFrame con columnas esperadas o levanta excepción.
    """
    df = fetch_meteobahia_api_xml(url.strip(), token=token or None, use_browser_headers=use_browser_headers)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError("La API no devolvió una tabla válida.")
    # Mapear columnas mínimas
    cols_lower = {c.lower(): c for c in df.columns}
    mapeo = {}
    for cands, tgt in [
        (["fecha", "date", "time"], "Fecha"),
        (["tmax", "tx", "tempmax"], "TMAX"),
        (["tmin", "tn", "tempmin"], "TMIN"),
        (["prec", "ppt", "rain", "lluvia", "mm"], "Prec"),
    ]:
        for c in cands:
            if c in cols_lower:
                mapeo[cols_lower[c]] = tgt
                break
    df = df.rename(columns=mapeo)

    if "Fecha" not in df.columns:
        raise RuntimeError("La respuesta de la API no contiene columna de fecha reconocible.")
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.dropna(subset=["Fecha"])

    for c in ["TMAX", "TMIN", "Prec"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = [c for c in ["Fecha", "TMAX", "TMIN", "Prec"] if c in df.columns]
    return df[keep].sort_values("Fecha").reset_index(drop=True)

def normalize_hist(df_hist: pd.DataFrame, api_year: int) -> pd.DataFrame:
    """Normaliza histórico: acepta Fecha o solo Julian_days. Valida 1–365/366 y nombres variados."""
    import calendar
    df = df_hist.copy()

    df.columns = [str(c).strip() for c in df.columns]
    low2orig = {c.lower(): c for c in df.columns}

    ren = {}
    for cands, tgt in [
        (["fecha", "date", "fechas"], "Fecha"),
        (["julian_days", "julianday", "julian", "dia_juliano", "diajuliano"], "Julian_days"),
        (["tmax", "t_max", "t max", "tx", "tmax(°c)"], "TMAX"),
        (["tmin", "t_min", "t min", "tn", "tmin(°c)"], "TMIN"),
        (["prec", "ppt", "precip", "lluvia", "mm", "prcp"], "Prec"),
    ]:
        for c in cands:
            if c in low2orig:
                ren[low2orig[c]] = tgt
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
        jd = df["Julian_days"]
        df = df[(jd >= 1) & (jd <= max_j)].copy()
        if not df.empty:
            df["Julian_days"] = df["Julian_days"].astype(int)

    if "Fecha" not in df.columns and "Julian_days" in df.columns and not df.empty:
        base = pd.Timestamp(int(api_year), 1, 1)
        df["Fecha"] = df["Julian_days"].astype(int).apply(lambda d: base + pd.Timedelta(days=d - 1))

    if "Julian_days" not in df.columns and "Fecha" in df.columns:
        df["Julian_days"] = df["Fecha"].dt.dayofyear

    if "Fecha" in df.columns and not df.empty:
        df = df.loc[df["Fecha"].dt.year == int(api_year)].copy()

    req = {"Fecha", "Julian_days", "TMAX", "TMIN", "Prec"}
    if not req.issubset(df.columns):
        return pd.DataFrame()

    df = df.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    for c in ["TMAX", "TMIN", "Prec"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["Fecha", "Julian_days", "TMAX", "TMIN", "Prec"]]

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

    # Token (sin label visible)
    st.sidebar.text_input(
        label=" ",
        key="api_token",
        type="password",
        label_visibility="collapsed"
    )

    # Botón de recarga
    if st.sidebar.button("Actualizar ahora (forzar recarga)"):
        st.session_state["reload_nonce"] += 1

    token = st.session_state["api_token"] or ""
    compat = bool(st.session_state["compat_headers"])

    # API
    try:
        with st.spinner("Descargando pronóstico..."):
            df_api = fetch_api_cached(api_url, token, st.session_state["reload_nonce"], compat)
    except Exception as e:
        st.error(f"No se pudo cargar el pronóstico: {e}")
        df_api = pd.DataFrame()

    # Limitar a los primeros 8 días (si hay datos)
    if not df_api.empty:
        df_api["Fecha"] = pd.to_datetime(df_api["Fecha"])
        df_api = df_api.sort_values("Fecha")
        dias_unicos = df_api["Fecha"].dt.normalize().unique()
        df_api = df_api[df_api["Fecha"].dt.normalize().isin(dias_unicos[:8])]

    if df_api.empty:
        st.error("No se pudieron obtener datos del pronóstico.")
        st.stop()

    # Histórico (fijo, sin UI)
    dfh_raw = read_hist_from_url(DEFAULT_HIST_URL)

    # Fusión
    min_api_date = pd.to_datetime(df_api["Fecha"].min()).normalize()
    api_year = int(min_api_date.year)

    df_hist_trim = pd.DataFrame(columns=["Fecha", "Julian_days", "TMAX", "TMIN", "Prec"])
    if not dfh_raw.empty:
        try:
            df_hist_all = normalize_hist(dfh_raw, api_year=api_year)
            if not df_hist_all.empty:
                start_hist = pd.Timestamp(api_year, 1, 1)
                end_hist = min_api_date - pd.Timedelta(days=1)
                m = (df_hist_all["Fecha"] >= start_hist) & (df_hist_all["Fecha"] <= end_hist)
                df_hist_trim = df_hist_all.loc[m].copy()
                if df_hist_trim.empty:
                    st.warning(f"El histórico no aporta filas entre {start_hist.date()} y {end_hist.date()}.")
        except Exception as e:
            st.error(f"Error normalizando histórico: {e}")

    df_all = pd.concat([df_hist_trim, df_api], ignore_index=True)
    df_all["Fecha"] = pd.to_datetime(df_all["Fecha"], errors="coerce")
    df_all = df_all.dropna(subset=["Fecha"]).sort_values("Fecha")
    df_all = df_all.drop_duplicates(subset=["Fecha"], keep="last").reset_index(drop=True)
    df_all["Julian_days"] = df_all["Fecha"].dt.dayofyear

    if df_all.empty:
        st.error("Fusión vacía (ni histórico válido ni API).")
        st.stop()

    input_df_raw = df_all.copy()
    src = ["API"]
    if not df_hist_trim.empty:
        src.append(f"Hist ({df_hist_trim['Fecha'].min().date()} → {df_hist_trim['Fecha'].max().date()})")
    source_label = " + ".join(src)

elif fuente == "Subir Excel":
    uploaded_file = st.file_uploader("Cargar archivo input.xlsx", type=["xlsx"])
    if uploaded_file is not None:
        try:
            input_df_raw = pd.read_excel(uploaded_file)
            source_label = f"Excel: {uploaded_file.name}"
        except Exception as e:
            st.error(f"No pude leer el Excel: {e}")

# ================= Validación de entrada =================
if input_df_raw is None or input_df_raw.empty:
    st.stop()

# ================= Preparar datos p/ modelo =================
input_df = preparar_para_modelo(input_df_raw)
if input_df is None or input_df.empty:
    st.error("Tras preparar columnas, no quedaron filas válidas (julian_days, TMAX, TMIN, Prec).")
    st.stop()

# ================= Pesos del modelo =================
try:
    IW = np.load("IW.npy")
    bias_IW = np.load("bias_IW.npy")
    LW = np.load("LW.npy")
    bias_out = np.load("bias_out.npy")
except Exception as e:
    st.error(f"No pude cargar los pesos del modelo (.npy): {e}")
    st.stop()

# ================= Ejecutar modelo =================
try:
    resultado = ejecutar_modelo(input_df, IW, bias_IW, LW, bias_out, umbral_usuario)
except TypeError as e:
    st.error(f"No se pudo ejecutar el modelo. Detalle: {e}")
    st.stop()

# Reemplazar Fecha por la del input original si está completa
fechas_excel = usar_fechas_de_input(input_df_raw, len(resultado))
if fechas_excel is not None:
    resultado["Fecha"] = fechas_excel

# ================= Rango 1-feb → 1-oct =================
pred_vis = reiniciar_feb_oct(resultado[["Fecha", "EMERREL (0-1)"]].copy(), umbral_ajustable=umbral_usuario)

# Sello y fuente (sin exponer URL)
if source_label:
    st.caption(f"Fuente de datos: {source_label}")
st.caption(f"Última actualización: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption(f"Umbral EMEAC usado: {umbral_usuario}" + (" (forzado desde código)" if usar_codigo else ""))

# ================= Gráficos + Tabla (rango 1-feb → 1-oct) =================
if not pred_vis.empty:
    pred_vis = pred_vis.copy()
    pred_vis["EMERREL_MA5_rango"] = pred_vis["EMERREL (0-1)"].rolling(5, min_periods=1).mean()

    # Clasificación por niveles (0.2 / 0.4)
    def clasif(v): return "Bajo" if v < 0.2 else ("Medio" if v < 0.4 else "Alto")
    pred_vis["Nivel de EMERREL"] = pred_vis["EMERREL (0-1)"].apply(clasif)

    # (OPCIONAL) Forzar "Bajo" cuando EMEAC < 10% (descomentar si lo querés)
    # emeac_cumsum = np.cumsum(pred_vis["EMERREL (0-1)"].to_numpy())
    # emeac_pct = np.clip(emeac_cumsum / float(umbral_usuario) * 100.0, 0, 100)
    # pred_vis.loc[emeac_pct < 10, "Nivel de EMERREL"] = "Bajo"

    # ---------- SERIES EMEAC ----------
    emerrel_rango = pred_vis["EMERREL (0-1)"].to_numpy()
    cumsum_rango = np.cumsum(emerrel_rango)
    emeac_min_pct = np.clip(cumsum_rango / float(EMEAC_MAX) * 100.0, 0, 100)  # inferior
    emeac_max_pct = np.clip(cumsum_rango / float(EMEAC_MIN) * 100.0, 0, 100)  # superior
    emeac_ajust   = np.clip(cumsum_rango / float(umbral_usuario) * 100.0, 0, 100)

    # === Plot con Plotly si está disponible ===
    if PLOTLY_OK:
        color_map = {"Bajo": "green", "Medio": "yellow", "Alto": "red"}

        # ---------- Gráfico 1: EMERREL ----------
        st.subheader("EMERGENCIA RELATIVA DIARIA - BORDENAVE")
        fig1 = go.Figure()

        fig1.add_bar(
            x=pred_vis["Fecha"],
            y=pred_vis["EMERREL (0-1)"],
            marker=dict(color=pred_vis["Nivel de EMERREL"].map(color_map).tolist()),
            customdata=pred_vis["Nivel de EMERREL"],
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>EMERREL: %{y:.3f}<br>Nivel: %{customdata}<extra></extra>",
            name="EMERREL (0-1)",
        )
        fig1.add_trace(go.Scatter(
            x=pred_vis["Fecha"], y=pred_vis["EMERREL_MA5_rango"],
            mode="lines", name="Media móvil 5 días",
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>MA5: %{y:.3f}<extra></extra>"
        ))
        fig1.add_trace(go.Scatter(
            x=pred_vis["Fecha"], y=pred_vis["EMERREL_MA5_rango"],
            mode="lines", line=dict(width=0),
            fill="tozeroy", fillcolor="rgba(135, 206, 250, 0.3)",
            name="Área MA5", hoverinfo="skip", showlegend=False
        ))

        y_low, y_med = 0.2, 0.4
        x0, x1 = pred_vis["Fecha"].min(), pred_vis["Fecha"].max()
        fig1.add_trace(go.Scatter(
            x=[x0, x1], y=[y_low, y_low],
            mode="lines", line=dict(color="green", dash="dot"),
            name=f"Nivel Bajo (≤ {y_low:.2f})", hoverinfo="skip"
        ))
        fig1.add_trace(go.Scatter(
            x=[x0, x1], y=[y_med, y_med],
            mode="lines", line=dict(color="orange", dash="dot"),
            name=f"Nivel Medio (≤ {y_med:.2f})", hoverinfo="skip"
        ))
        fig1.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color="red", dash="dot"),
            name=f"Nivel Alto (> {y_med:.2f})", hoverinfo="skip"
        ))

        fig1.update_layout(
            xaxis_title="Fecha", yaxis_title="EMERREL (0-1)",
            hovermode="x unified",
            legend_title="Referencias",
            height=650
        )
        st.plotly_chart(fig1, use_container_width=True, theme="streamlit")

        # ---------- Gráfico 2: EMEAC ----------
        st.subheader("EMERGENCIA ACUMULADA DIARIA - BORDENAVE")
        st.markdown(f"**Umbrales:** Min={EMEAC_MIN} · Max={EMEAC_MAX} · Ajustable={umbral_usuario}")

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=pred_vis["Fecha"], y=emeac_min_pct,
            mode="lines", line=dict(width=0),
            name=f"Mínimo (umbral {EMEAC_MAX})",
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Mínimo: %{y:.1f}%<extra></extra>"
        ))
        fig2.add_trace(go.Scatter(
            x=pred_vis["Fecha"], y=emeac_max_pct,
            mode="lines", line=dict(width=0),
            fill="tonexty",
            name=f"Máximo (umbral {EMEAC_MIN})",
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Máximo: %{y:.1f}%<extra></extra>"
        ))
        fig2.add_trace(go.Scatter(
            x=pred_vis["Fecha"], y=emeac_ajust,
            mode="lines", name=f"Ajustable ({umbral_usuario})",
            line=dict(width=2.5),
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Ajustable: %{y:.1f}%<extra></extra>"
        ))
        fig2.add_trace(go.Scatter(
            x=pred_vis["Fecha"], y=emeac_min_pct,
            mode="lines", name=f"Mínimo (umbral {EMEAC_MAX})",
            line=dict(dash="dash", width=1.5),
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Mínimo: %{y:.1f}%<extra></extra>"
        ))
        fig2.add_trace(go.Scatter(
            x=pred_vis["Fecha"], y=emeac_max_pct,
            mode="lines", name=f"Máximo (umbral {EMEAC_MIN})",
            line=dict(dash="dash", width=1.5),
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Máximo: %{y:.1f}%<extra></extra>"
        ))

        for nivel in [25, 50, 75, 90]:
            fig2.add_hline(y=nivel, line_dash="dash", opacity=0.6, annotation_text=f"{nivel}%")

        fig2.update_layout(
            xaxis_title="Fecha", yaxis_title="EMEAC (%)",
            hovermode="x unified",
            legend_title="Referencias",
            yaxis=dict(range=[0, 100]),
            height=600
        )
        st.plotly_chart(fig2, use_container_width=True, theme="streamlit")

    else:
        # === Fallback Matplotlib ===
        color_map = {"Bajo": "green", "Medio": "yellow", "Alto": "red"}

        # --- Gráfico 1: EMERREL ---
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        ax1.fill_between(
            pred_vis["Fecha"], 0, pred_vis["EMERREL_MA5_rango"],
            color="skyblue", alpha=0.3, zorder=0
        )
        ax1.bar(
            pred_vis["Fecha"], pred_vis["EMERREL (0-1)"],
            color=pred_vis["Nivel de EMERREL"].map(color_map)
        )
        line_ma5 = ax1.plot(
            pred_vis["Fecha"], pred_vis["EMERREL_MA5_rango"],
            linewidth=2.2, label="Media móvil 5 días"
        )[0]
        ax1.set_ylabel("EMERREL (0-1)")
        ax1.set_title("EMERGENCIA RELATIVA DIARIA - BORDENAVE")
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(
            handles=[Patch(facecolor=color_map[k], label=k) for k in ["Bajo","Medio","Alto"]] + [line_ma5],
            loc="upper right"
        )
        ax1.grid(True)
        st.pyplot(fig1); plt.close(fig1)

        # --- Gráfico 2: EMEAC ---
        st.subheader("EMERGENCIA ACUMULADA DIARIA - BORDENAVE")
        st.markdown(f"**Umbrales:** Min={EMEAC_MIN} · Max={EMEAC_MAX} · Ajustable={umbral_usuario}")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(pred_vis["Fecha"], emeac_ajust,   label=f"Ajustable ({umbral_usuario})", linewidth=2)
        ax2.plot(pred_vis["Fecha"], emeac_min_pct, label=f"Mínimo (umbral {EMEAC_MAX})", linestyle="--", linewidth=2)
        ax2.plot(pred_vis["Fecha"], emeac_max_pct, label=f"Máximo (umbral {EMEAC_MIN})", linestyle="--", linewidth=2)
        ax2.fill_between(pred_vis["Fecha"], emeac_min_pct, emeac_max_pct, alpha=0.3, label="Área entre Mín y Máx")
        ax2.set_ylabel("EMEAC (%)")
        ax2.set_ylim(0, 105)
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2); plt.close(fig2)

    # --- Tabla (después de los gráficos) ---
    pred_vis["Día juliano"] = pd.to_datetime(pred_vis["Fecha"]).dt.dayofyear
    nivel_emoji = {"Bajo": "🟢", "Medio": "🟡", "Alto": "🔴"}
    nivel_emoji_txt = pred_vis["Nivel de EMERREL"].map(lambda x: f"{nivel_emoji.get(x, '')} {x}")

    tabla_display = pd.DataFrame({
        "Fecha": pred_vis["Fecha"],
        "Día juliano": pred_vis["Día juliano"].astype(int),
        "Nivel de EMERREL": nivel_emoji_txt,
        "EMEAC (%)": emeac_ajust
    })

    tabla_csv = pd.DataFrame({
        "Fecha": pred_vis["Fecha"],
        "Día juliano": pred_vis["Día juliano"].astype(int),
        "Nivel de EMERREL": pred_vis["Nivel de EMERREL"],
        "EMEAC (%)": emeac_ajust
    })

    st.subheader("Tabla de Resultados (rango 1-feb → 1-oct)")
    st.dataframe(tabla_display, use_container_width=True)

    csv_rango = tabla_csv.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar tabla (rango) en CSV",
        data=csv_rango,
        file_name=f"tabla_rango_{pd.Timestamp.now().strftime('%Y-%m-%d_%H%M')}.csv",
        mime="text/csv",
    )
else:
    st.warning("No hay datos en el rango 1-feb → 1-oct para el año detectado.")
