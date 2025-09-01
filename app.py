# app.py — PREDICCION EMERGENCIA AGRICOLA HIRSHIN (con persistencia de pronóstico vencido → histórico)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import io
import requests
import base64
import json
from datetime import datetime, timezone

from modelo_emerrel import ejecutar_modelo
from meteobahia import (
    preparar_para_modelo,
    usar_fechas_de_input,
    reiniciar_feb_oct,
)
from meteobahia_api import fetch_meteobahia_api_xml  # usa headers tipo navegador
from fetch_meteobahia import load_public_csv  # carga CSV público validado

# ==== Plotly (opcional) ====
try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

st.set_page_config(page_title="PREDICCION EMERGENCIA AGRICOLA HIRSHIN", layout="wide")

# ====================== UMBRALES EMEAC (EDITABLES EN CÓDIGO) ======================
EMEAC_MIN = 5     # Umbral mínimo por defecto (cambia aquí)
EMEAC_MAX = 7     # Umbral máximo por defecto (cambia aquí)
EMEAC_MIN, EMEAC_MAX = sorted([EMEAC_MIN, EMEAC_MAX])

# Umbral AJUSTABLE por defecto (editable en CÓDIGO) y opción de forzarlo
EMEAC_AJUSTABLE_DEF = 6                 # Debe estar entre EMEAC_MIN y EMEAC_MAX
FORZAR_AJUSTABLE_DESDE_CODIGO = False   # True = ignora el slider y usa EMEAC_AJUSTABLE_DEF

# ====================== Config fija (no visible) ======================
DEFAULT_API_URL = "https://meteobahia.com.ar/scripts/forecast/for-bd.xml"  # NUNCA visible en UI

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
    options=["API + Histórico (CSV público)", "Subir Excel"],
    index=0,
)
usar_codigo = st.sidebar.checkbox(
    label=" ",
    value=FORZAR_AJUSTABLE_DESDE_CODIGO,
    key="chk_usar_codigo",
    label_visibility="collapsed"
)

umbral_slider = st.sidebar.slider(
    "Seleccione el umbral EMEAC (Ajustable)",
    min_value=int(EMEAC_MIN),
    max_value=int(EMEAC_MAX),
    value=int(np.clip(EMEAC_AJUSTABLE_DEF, EMEAC_MIN, EMEAC_MAX))
)

# Umbral efectivo que usa la app
umbral_usuario = int(np.clip(
    EMEAC_AJUSTABLE_DEF if usar_codigo else umbral_slider,
    EMEAC_MIN, EMEAC_MAX
))

# ============== Helpers API cache =================
@st.cache_data(ttl=600)
def fetch_api_cached(url: str, token: str | None, nonce: int, use_browser_headers: bool):
    # 'nonce' invalida la caché
    return fetch_meteobahia_api_xml(url.strip(), token=token or None, use_browser_headers=use_browser_headers)

def read_hist_from_url(url: str) -> pd.DataFrame:
    """Compat: si quisieras leer Excel/CSV arbitrario desde URL (no se usa en flujo principal)."""
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

# ===================== Persistencia: GitHub (CSV público) =====================

def _github_headers():
    return {
        "Authorization": f"Bearer {st.secrets['GH_TOKEN']}",
        "Accept": "application/vnd.github+json",
    }

def _github_get_file_sha(repo, path, ref):
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={ref}"
    r = requests.get(url, headers=_github_headers(), timeout=30)
    if r.status_code == 200:
        return r.json().get("sha")
    elif r.status_code == 404:
        return None
    r.raise_for_status()

def _github_put_file(repo, path, branch, content_bytes, msg, sha=None, committer=None):
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    payload = {
        "message": msg,
        "content": base64.b64encode(content_bytes).decode("utf-8"),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha
    if committer:
        payload["committer"] = committer
    r = requests.put(url, headers=_github_headers(), data=json.dumps(payload), timeout=30)
    r.raise_for_status()
    return r.json()

def promote_forecast_into_history(df_hist: pd.DataFrame, df_api: pd.DataFrame) -> pd.DataFrame:
    """
    Promueve del pronóstico solo fechas <= 'hoy' (zona America/Argentina/Buenos_Aires).
    Prioridad por Fecha: Histórico (medido/guardado) > Pronóstico.
    Devuelve nuevo histórico continuo.
    """
    if df_hist is None or df_hist.empty:
        df_hist = pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    else:
        df_hist = df_hist.copy()

    df_hist["Fecha"] = pd.to_datetime(df_hist["Fecha"], errors="coerce")

    if df_api is None or df_api.empty:
        return df_hist.sort_values("Fecha").reset_index(drop=True)

    df_api = df_api.copy()
    df_api["Fecha"] = pd.to_datetime(df_api["Fecha"], errors="coerce")

    hoy_local = pd.Timestamp.now(tz="America/Argentina/Buenos_Aires").normalize()
    vencido = df_api.loc[df_api["Fecha"].dt.tz_localize(None) <= hoy_local.tz_localize(None)]

    if vencido.empty:
        return df_hist.sort_values("Fecha").reset_index(drop=True)

    # Concat histórico primero (prioridad), luego pronóstico vencido; quitar duplicados por Fecha quedándose con el primero
    merged = pd.concat([df_hist.sort_values("Fecha"), vencido], ignore_index=True)
    merged = merged.dropna(subset=["Fecha"]).sort_values(["Fecha"]).drop_duplicates(subset=["Fecha"], keep="first")

    merged["Julian_days"] = merged["Fecha"].dt.dayofyear
    for c in ["TMAX","TMIN","Prec"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    return merged.sort_values("Fecha").reset_index(drop=True)

def try_commit_history_csv(df_hist_nuevo: pd.DataFrame) -> bool:
    """Sube el CSV actualizado al repo configurado en secrets. True si comiteó."""
    repo   = st.secrets["GH_REPO"]
    branch = st.secrets["GH_BRANCH"]
    path   = st.secrets["GH_PATH"]

    sha_actual = _github_get_file_sha(repo, path, branch)
    csv_bytes = df_hist_nuevo.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")

    committer = None
    if "GH_USER_NAME" in st.secrets and "GH_USER_EMAIL" in st.secrets:
        committer = {"name": st.secrets["GH_USER_NAME"], "email": st.secrets["GH_USER_EMAIL"]}

    ahora_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    msg = f"[auto] Promover pronóstico vencido a histórico ({ahora_utc})"

    _github_put_file(repo, path, branch, csv_bytes, msg, sha=sha_actual, committer=committer)
    return True

# ================= Flujo principal =================
st.title("PREDICCION EMERGENCIA AGRICOLA HIRSHIN")

input_df_raw = None
source_label = None

if fuente == "API + Histórico (CSV público)":
    # Pronóstico (API XML) — URL NUNCA visible
    api_url = DEFAULT_API_URL  # fija y oculta (solo en código)

    # Token (opcional) y headers compat
    st.sidebar.text_input(label=" ", key="api_token", type="password", label_visibility="collapsed")
    st.session_state["compat_headers"] = st.sidebar.checkbox(
        "Compatibilidad (headers de navegador)", value=st.session_state["compat_headers"]
    )
    if st.sidebar.button("Actualizar ahora (forzar recarga)"):
        st.session_state["reload_nonce"] += 1

    token = st.session_state["api_token"] or ""
    compat = bool(st.session_state["compat_headers"])

    # === 1) API (limitada a 8 días) ============================================
    with st.spinner("Descargando pronóstico..."):
        df_api = fetch_api_cached(api_url, token, st.session_state["reload_nonce"], compat)
    df_api["Fecha"] = pd.to_datetime(df_api["Fecha"])
    df_api = df_api.sort_values("Fecha")
    dias_unicos = df_api["Fecha"].dt.normalize().unique()
    df_api = df_api[df_api["Fecha"].dt.normalize().isin(dias_unicos[:8])]

    if df_api.empty:
        st.error("No se pudieron obtener datos del pronóstico (API).")
        st.stop()

    # === 2) Cargar histórico público actual (CSV validado) =====================
    try:
        df_hist_publico, _hist_src = load_public_csv(parse_dates=True)  # valida columnas/fechas
        hist_src_lbl = "CSV público (Pages/Raw)"
    except Exception as e:
        st.warning(f"No pude leer el histórico público (persistencia desactivada en esta ejecución): {e}")
        df_hist_publico = pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
        hist_src_lbl = "CSV público (no disponible)"

    # === 3) Promover pronóstico vencido → histórico y commitear ================
    commit_msg = None
    try:
        df_hist_actualizado = promote_forecast_into_history(df_hist_publico, df_api)

        # ¿Cambió algo? (largo o última fecha)
        commit_needed = (
            len(df_hist_actualizado) != len(df_hist_publico)
            or (not df_hist_actualizado.empty and not df_hist_publico.empty
                and df_hist_actualizado["Fecha"].max() != df_hist_publico["Fecha"].max())
        )

        if commit_needed and {"GH_TOKEN","GH_REPO","GH_BRANCH","GH_PATH"}.issubset(st.secrets):
            if try_commit_history_csv(df_hist_actualizado):
                commit_msg = "Histórico actualizado en el repositorio (pronóstico vencido promovido)."
                st.success(commit_msg)
        elif commit_needed:
            st.info("Cambios detectados en histórico, pero faltan secrets de GitHub para comitear automáticamente.")
    except Exception as e:
        st.warning(f"No se pudo comitear el histórico al repo: {e}")

    # === 4) Releer histórico publicado tras commit (si aplica) ==================
    try:
        df_hist_publico2, _ = load_public_csv(parse_dates=True)
    except Exception:
        df_hist_publico2 = df_hist_publico.copy()

    # === 5) Fusión para la app (permitiendo solape en el límite) ===============
    min_api_date = pd.to_datetime(df_api["Fecha"].min()).normalize()
    api_year = int(min_api_date.year)
    start_hist = pd.Timestamp(api_year, 1, 1)
    end_hist = min_api_date  # permitir solape para no cortar agosto el día previo
    mask = (df_hist_publico2["Fecha"] >= start_hist) & (df_hist_publico2["Fecha"] <= end_hist)
    df_hist_trim = df_hist_publico2.loc[mask].copy()

    df_all = pd.concat([df_hist_trim, df_api], ignore_index=True)
    df_all["Fecha"] = pd.to_datetime(df_all["Fecha"], errors="coerce")
    df_all = df_all.dropna(subset=["Fecha"]).sort_values("Fecha").drop_duplicates(subset=["Fecha"], keep="last").reset_index(drop=True)
    df_all["Julian_days"] = df_all["Fecha"].dt.dayofyear

    if df_all.empty:
        st.error("Fusión vacía (ni histórico válido ni API).")
        st.stop()

    input_df_raw = df_all.copy()
    src = [f"API (8 días)"]
    if not df_hist_trim.empty:
        src.append(f"Hist (CSV público: {df_hist_trim['Fecha'].min().date()} → {df_hist_trim['Fecha'].max().date()})")
    source_label = " + ".join(src)

elif fuente == "Subir Excel":
    uploaded_file = st.file_uploader("Cargar archivo input.xlsx", type=["xlsx", "csv"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                input_df_raw = pd.read_csv(uploaded_file)
            else:
                input_df_raw = pd.read_excel(uploaded_file)
            source_label = f"Excel/CSV: {uploaded_file.name}"
        except Exception as e:
            st.error(f"No pude leer el archivo: {e}")

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
resultado = ejecutar_modelo(input_df, IW, bias_IW, LW, bias_out, umbral_usuario)

# Reemplazar Fecha por la del input original si está completa
fechas_excel = usar_fechas_de_input(input_df_raw, len(resultado))
if fechas_excel is not None:
    resultado["Fecha"] = fechas_excel

# ================= Rango 1-feb → 1-oct =================
pred_vis = reiniciar_feb_oct(resultado[["Fecha", "EMERREL (0-1)"]].copy(), umbral_ajustable=umbral_usuario)

# Sello y fuente (sin exponer URL)
st.caption(f"Fuente de datos: {source_label or '—'}")
st.caption(f"Última actualización: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption(f"Umbral EMEAC usado: {umbral_usuario}" + (" (forzado desde código)" if usar_codigo else ""))

# ================= Gráficos + Tabla (rango 1-feb → 1-oct) =================
if not pred_vis.empty:
    pred_vis = pred_vis.copy()
    pred_vis["EMERREL_MA5_rango"] = pred_vis["EMERREL (0-1)"].rolling(5, min_periods=1).mean()

    def clasif(v): return "Bajo" if v < 0.2 else ("Medio" if v < 0.4 else "Alto")
    pred_vis["Nivel de EMERREL"] = pred_vis["EMERREL (0-1)"].apply(clasif)

    emerrel_rango = pred_vis["EMERREL (0-1)"].to_numpy()
    cumsum_rango = np.cumsum(emerrel_rango)
    emeac_min_pct = np.clip(cumsum_rango / float(EMEAC_MAX) * 100.0, 0, 100)  # min = umbral más alto
    emeac_max_pct = np.clip(cumsum_rango / float(EMEAC_MIN) * 100.0, 0, 100)  # max = umbral más bajo
    emeac_ajust   = np.clip(cumsum_rango / float(umbral_usuario) * 100.0, 0, 100)

    if PLOTLY_OK:
        color_map = {"Bajo": "green", "Medio": "yellow", "Alto": "red"}

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
        fig1.update_layout(xaxis_title="Fecha", yaxis_title="EMERREL (0-1)", hovermode="x unified",
                           legend_title="Referencias", height=650)
        st.plotly_chart(fig1, use_container_width=True, theme="streamlit")

        st.subheader("EMERGENCIA ACUMULADA DIARIA - BORDENAVE")
        st.markdown(f"**Umbrales:** Min={EMEAC_MIN} · Max={EMEAC_MAX} · Ajustable={umbral_usuario}")

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=pred_vis["Fecha"], y=emeac_min_pct, mode="lines", line=dict(width=0),
                                  name=f"Mínimo (umbral {EMEAC_MAX})",
                                  hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Mínimo: %{y:.1f}%<extra></extra>"))
        fig2.add_trace(go.Scatter(x=pred_vis["Fecha"], y=emeac_max_pct, mode="lines", line=dict(width=0),
                                  fill="tonexty", name=f"Máximo (umbral {EMEAC_MIN})",
                                  hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Máximo: %{y:.1f}%<extra></extra>"))
        fig2.add_trace(go.Scatter(x=pred_vis["Fecha"], y=emeac_ajust, mode="lines",
                                  name=f"Ajustable ({umbral_usuario})", line=dict(width=2.5),
                                  hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Ajustable: %{y:.1f}%<extra></extra>"))
        fig2.add_trace(go.Scatter(x=pred_vis["Fecha"], y=emeac_min_pct, mode="lines",
                                  name=f"Mínimo (umbral {EMEAC_MAX})", line=dict(dash="dash", width=1.5),
                                  hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Mínimo: %{y:.1f}%<extra></extra>"))
        fig2.add_trace(go.Scatter(x=pred_vis["Fecha"], y=emeac_max_pct, mode="lines",
                                  name=f"Máximo (umbral {EMEAC_MIN})", line=dict(dash="dash", width=1.5),
                                  hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Máximo: %{y:.1f}%<extra></extra>"))
        for nivel in [25, 50, 75, 90]:
            fig2.add_hline(y=nivel, line_dash="dash", opacity=0.6, annotation_text=f"{nivel}%")

        fig2.update_layout(xaxis_title="Fecha", yaxis_title="EMEAC (%)", hovermode="x unified",
                           legend_title="Referencias", yaxis=dict(range=[0, 100]), height=600)
        st.plotly_chart(fig2, use_container_width=True, theme="streamlit")

    else:
        # Fallback Matplotlib
        color_map = {"Bajo": "green", "Medio": "yellow", "Alto": "red"}
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        ax1.fill_between(pred_vis["Fecha"], 0, pred_vis["EMERREL_MA5_rango"], color="skyblue", alpha=0.3, zorder=0)
        ax1.bar(pred_vis["Fecha"], pred_vis["EMERREL (0-1)"], color=pred_vis["Nivel de EMERREL"].map(color_map))
        line_ma5 = ax1.plot(pred_vis["Fecha"], pred_vis["EMERREL_MA5_rango"], linewidth=2.2, label="Media móvil 5 días")[0]
        ax1.set_ylabel("EMERREL (0-1)")
        ax1.set_title("EMERGENCIA RELATIVA DIARIA")
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(handles=[Patch(facecolor=color_map[k], label=k) for k in ["Bajo","Medio","Alto"]] + [line_ma5], loc="upper right")
        ax1.grid(True)
        st.pyplot(fig1); plt.close(fig1)

        st.subheader("EMERGENCIA ACUMULADA DIARIA")
        st.markdown(f"**Umbrales:** Min={EMEAC_MIN} · Max={EMEAC_MAX} · Ajustable={umbral_usuario}")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(pred_vis["Fecha"], emeac_ajust,   label=f"Ajustable ({umbral_usuario})", linewidth=2)
        ax2.plot(pred_vis["Fecha"], emeac_min_pct, label=f"Mínimo (umbral {EMEAC_MAX})", linestyle="--", linewidth=2)
        ax2.plot(pred_vis["Fecha"], emeac_max_pct, label=f"Máximo (umbral {EMEAC_MIN})", linestyle="--", linewidth=2)
        ax2.fill_between(pred_vis["Fecha"], emeac_min_pct, emeac_max_pct, alpha=0.3, label="Área entre Mín y Máx")
        ax2.set_ylabel("EMEAC (%)"); ax2.set_ylim(0, 105); ax2.legend(); ax2.grid(True)
        st.pyplot(fig2); plt.close(fig2)

    # --- Tabla ---------------------------------------------------------------
    pred_vis["Día juliano"] = pd.to_datetime(pred_vis["Fecha"]).dt.dayofyear
    nivel_emoji = {"Bajo": "🟢", "Medio": "🟡", "Alto": "🔴"}
    nivel_emoji_txt = pred_vis["Nivel de EMERREL"].map(lambda x: f"{nivel_emoji.get(x, '')} {x}")

    tabla_display = pd.DataFrame({
        "Fecha": pred_vis["Fecha"],
        "Día juliano": pred_vis["Día juliano"].astype(int),
        "Nivel de EMERREL": nivel_emoji_txt,
        "EMEAC (%)": np.clip(np.cumsum(pred_vis["EMERREL (0-1)"]) / float(umbral_usuario) * 100.0, 0, 100)
    })
    tabla_csv = pd.DataFrame({
        "Fecha": pred_vis["Fecha"],
        "Día juliano": pred_vis["Día juliano"].astype(int),
        "Nivel de EMERREL": pred_vis["Nivel de EMERREL"],
        "EMEAC (%)": np.clip(np.cumsum(pred_vis["EMERREL (0-1)"]) / float(umbral_usuario) * 100.0, 0, 100)
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

