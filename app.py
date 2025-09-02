# app.py – HIRSHIN (serie completa 1-feb → 1-oct 2025; usa TODO el pronóstico API; sin exclusiones)
import os
import io
import json
import base64
from datetime import datetime, timezone

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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

# ====================== VENTANA FIJA (VISUALIZACION) ======================
FECHA_INICIO_FIJA = pd.Timestamp("2025-02-01")
FECHA_FIN_FIJA    = pd.Timestamp("2025-10-01")

# ====================== UMBRALES EMEAC (EDITABLES EN CÓDIGO) ======================
EMEAC_MIN = 5     # umbral mínimo
EMEAC_MAX = 7     # umbral máximo
EMEAC_MIN, EMEAC_MAX = sorted([EMEAC_MIN, EMEAC_MAX])

EMEAC_AJUSTABLE_DEF = 6                 # dentro de [EMEAC_MIN, EMEAC_MAX]
FORZAR_AJUSTABLE_DESDE_CODIGO = False   # True => ignora slider y usa EMEAC_AJUSTABLE_DEF

# === Regla de lluvia 7 días para clasificar EMERREL (NO excluye días; solo clasifica) ===
APLICAR_REGLA_LLUVIA_7D = True
LLUVIA_CORTE_MM_7D = 10.0   # inclusivo: se cumple con ≥10 mm
LLUVIA_VENTANA_DIAS = 7     # 7 días calendario previos (sin incluir el día actual)

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

# ===================== Helpers API/Histórico =====================
@st.cache_data(ttl=600)
def fetch_api_cached(url: str, token: str | None, nonce: int, use_browser_headers: bool):
    # 'nonce' invalida la caché
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

# ===================== Persistencia en GitHub (opcional) =====================
def _have_gh_secrets():
    req = ["GH_TOKEN", "GH_REPO", "GH_BRANCH", "GH_PATH"]
    return all(k in st.secrets for k in req)

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

def _normalize_hist_like(df: pd.DataFrame, api_year: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    low2orig = {c.lower(): c for c in out.columns}
    def has(c): return c in low2orig
    def col(c): return low2orig[c]
    ren = {}
    for cands, tgt in [
        (["fecha","date","fechas"], "Fecha"),
        (["julian_days","julianday","julian","dia_juliano","doy"], "Julian_days"),
        (["tmax","t_max","t max","tx","tmax(°c)"], "TMAX"),
        (["tmin","t_min","t min","tn","tmin(°c)"], "TMIN"),
        (["prec","ppt","precip","lluvia","mm","prcp"], "Prec"),
    ]:
        for c in cands:
            if has(c):
                ren[col(c)] = tgt
                break
    out = out.rename(columns=ren)
    if "Fecha" in out.columns:
        out["Fecha"] = pd.to_datetime(out["Fecha"], errors="coerce")
    for c in ["TMAX","TMIN","Prec","Julian_days"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "Fecha" not in out.columns and "Julian_days" in out.columns:
        base = pd.Timestamp(int(api_year), 1, 1)
        out["Fecha"] = out["Julian_days"].astype(float).apply(lambda d: base + pd.Timedelta(days=int(d) - 1))
    if "Julian_days" not in out.columns and "Fecha" in out.columns:
        out["Julian_days"] = pd.to_datetime(out["Fecha"]).dt.dayofyear
    req = {"Fecha","Julian_days","TMAX","TMIN","Prec"}
    faltan = req - set(out.columns)
    if faltan:
        return out
    out = out.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    out["Julian_days"] = pd.to_datetime(out["Fecha"]).dt.dayofyear
    for c in ["TMAX","TMIN","Prec"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out[["Fecha","Julian_days","TMAX","TMIN","Prec"]]

def promote_forecast_into_history(df_hist: pd.DataFrame, df_api: pd.DataFrame) -> pd.DataFrame:
    api_year = int(pd.to_datetime(df_api["Fecha"]).min().year) if (df_api is not None and not df_api.empty) else pd.Timestamp.now().year
    if df_hist is None:
        df_hist = pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    df_hist_norm = _normalize_hist_like(df_hist, api_year=api_year)
    if "Fecha" not in df_hist_norm.columns:
        st.warning(f"El histórico no contiene columna 'Fecha' tras normalización. Columnas encontradas: {list(df_hist.columns)}")
        df_hist_norm = pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    if df_api is None or df_api.empty:
        return df_hist_norm.sort_values("Fecha").reset_index(drop=True)
    df_api = df_api.copy()
    df_api["Fecha"] = pd.to_datetime(df_api["Fecha"], errors="coerce")
    hoy_local = pd.Timestamp.now(tz="America/Argentina/Buenos_Aires").normalize()
    vencido = df_api.loc[df_api["Fecha"].dt.tz_localize(None) <= hoy_local.tz_localize(None)]
    if vencido.empty:
        return df_hist_norm.sort_values("Fecha").reset_index(drop=True)
    merged = pd.concat([df_hist_norm.sort_values("Fecha"), vencido], ignore_index=True)
    merged = (merged.dropna(subset=["Fecha"]).sort_values(["Fecha"])
                   .drop_duplicates(subset=["Fecha"], keep="first").reset_index(drop=True))
    merged["Julian_days"] = merged["Fecha"].dt.dayofyear
    for c in ["TMAX","TMIN","Prec"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
    return merged.sort_values("Fecha").reset_index(drop=True)

def try_commit_history_csv(df_hist_nuevo: pd.DataFrame) -> bool:
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

# ================= Sidebar =================
st.sidebar.header("Fuente de datos")
fuente = st.sidebar.radio("Elegí cómo cargar datos", options=["API + Histórico", "Subir Excel"], index=0)
usar_codigo = st.sidebar.checkbox(label=" ", value=FORZAR_AJUSTABLE_DESDE_CODIGO, key="chk_usar_codigo", label_visibility="collapsed")
umbral_slider = st.sidebar.slider("Seleccione el umbral EMEAC (Ajustable)", min_value=int(EMEAC_MIN), max_value=int(EMEAC_MAX), value=int(np.clip(EMEAC_AJUSTABLE_DEF, EMEAC_MIN, EMEAC_MAX)))
umbral_usuario = int(np.clip(EMEAC_AJUSTABLE_DEF if usar_codigo else umbral_slider, EMEAC_MIN, EMEAC_MAX))

# ================= Flujo principal =================
st.title("PREDICCION EMERGENCIA AGRICOLA HIRSHIN")

input_df_raw = None
source_label = None

if fuente == "API + Histórico":
    api_url = DEFAULT_API_URL
    st.sidebar.text_input(label=" ", key="api_token", type="password", label_visibility="collapsed")
    st.session_state["compat_headers"] = st.sidebar.checkbox("Compatibilidad (headers de navegador)", value=st.session_state["compat_headers"])
    if st.sidebar.button("Actualizar ahora (forzar recarga)"):
        st.session_state["reload_nonce"] += 1
    token = st.session_state["api_token"] or ""
    compat = bool(st.session_state["compat_headers"])

    with st.spinner("Descargando pronóstico..."):
        df_api = fetch_api_cached(api_url, token, st.session_state["reload_nonce"], compat)

    # === usar TODOS los días del pronóstico (sin recorte); consolidar por día calendario ===
    df_api["Fecha"] = pd.to_datetime(df_api["Fecha"], errors="coerce")
    df_api = df_api.dropna(subset=["Fecha"]).sort_values("Fecha")
    if {"TMAX", "TMIN", "Prec"}.issubset(df_api.columns):
        df_api["Fecha"] = df_api["Fecha"].dt.normalize()
        df_api = (df_api
                  .groupby("Fecha", as_index=False)
                  .agg({"TMAX":"max", "TMIN":"min", "Prec":"sum"}))
    if df_api.empty:
        st.error("No se pudieron obtener datos del pronóstico.")
        st.stop()

    # --- Histórico (local/GitHub/URL) ---
    HIST_LOCAL = st.secrets.get("HIST_LOCAL_PATH", "").strip()
    candidatos = [p for p in [HIST_LOCAL, "./historico.xlsx", "/mnt/data/historico.xlsx"] if p]
    df_hist_publico = pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    hist_source_desc = "Hist (vacío)"
    for path in candidatos:
        if os.path.exists(path):
            try:
                df_hist_publico = pd.read_excel(path)
                hist_source_desc = f"Hist (local: {os.path.basename(path)})"
                break
            except Exception as e:
                st.warning(f"No pude leer el histórico local {path}: {e}")
    if df_hist_publico.empty:
        try:
            if _have_gh_secrets():
                from fetch_meteobahia import load_public_csv
                df_hist_publico, _ = load_public_csv(parse_dates=True)
                hist_source_desc = "Hist (GitHub público)"
            else:
                df_hist_publico = read_hist_from_url(DEFAULT_HIST_URL)
                hist_source_desc = "Hist (URL fija)"
        except Exception as e:
            st.warning(f"No pude leer el histórico público: {e}")
            df_hist_publico = pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
            hist_source_desc = "Hist (vacío)"

    df_hist_actualizado = promote_forecast_into_history(df_hist_publico, df_api)

    if _have_gh_secrets():
        try:
            commit_needed = (
                len(df_hist_actualizado) != len(df_hist_publico)
                or (not df_hist_actualizado.empty and not df_hist_publico.empty
                    and df_hist_actualizado["Fecha"].max() != df_hist_publico["Fecha"].max())
            )
            if commit_needed:
                if try_commit_history_csv(df_hist_actualizado):
                    st.success("Histórico actualizado en el repositorio (pronóstico vencido promovido).")
        except Exception as e:
            st.warning(f"No se pudo comitear el histórico al repo: {e}")

    df_hist_usable = df_hist_actualizado if not df_hist_actualizado.empty else df_hist_publico

    # --- Fusión ordenada ---
    min_api_date = pd.to_datetime(df_api["Fecha"].min()).normalize()
    api_year = int(min_api_date.year)
    start_hist = pd.Timestamp(api_year, 1, 1)
    end_hist = min_api_date

    df_hist_trim = pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    if not df_hist_usable.empty and end_hist >= start_hist:
        try:
            dfh = df_hist_usable.copy()
            dfh["Fecha"] = pd.to_datetime(dfh["Fecha"], errors="coerce")
            for c in ["TMAX","TMIN","Prec"]:
                if c in dfh.columns:
                    dfh[c] = pd.to_numeric(dfh[c], errors="coerce")
            m = (dfh["Fecha"] >= start_hist) & (dfh["Fecha"] <= end_hist)
            df_hist_trim = dfh.loc[m].copy()
            if df_hist_trim.empty:
                st.warning(f"El histórico no aporta filas entre {start_hist.date()} y {end_hist.date()}.")
        except Exception as e:
            st.error(f"Error preparando histórico para la fusión: {e}")

    df_all = pd.concat([df_hist_trim, df_api], ignore_index=True)
    df_all["Fecha"] = pd.to_datetime(df_all["Fecha"], errors="coerce")
    df_all = df_all.dropna(subset=["Fecha"]).sort_values("Fecha")
    df_all = df_all.drop_duplicates(subset=["Fecha"], keep="last").reset_index(drop=True)
    df_all["Julian_days"] = df_all["Fecha"].dt.dayofyear

    # === Lluvia acumulada 7 días previos (excluye día actual) – calendario ===
    df_prec_lluvia = df_all[["Fecha", "Prec"]].copy()
    df_prec_lluvia["Fecha"] = pd.to_datetime(df_prec_lluvia["Fecha"]).dt.normalize()
    df_prec_lluvia = df_prec_lluvia.groupby("Fecha", as_index=False)["Prec"].sum()
    idx_full_rain = pd.date_range(df_prec_lluvia["Fecha"].min(), df_prec_lluvia["Fecha"].max(), freq="D")
    s = (df_prec_lluvia.set_index("Fecha")["Prec"].reindex(idx_full_rain).fillna(0.0).astype(float))
    lluvia_7d_prev = s.shift(1).rolling(window=LLUVIA_VENTANA_DIAS, min_periods=LLUVIA_VENTANA_DIAS).sum().fillna(0.0)
    df_prec_lluvia = pd.DataFrame({"Fecha": idx_full_rain, "lluvia_7d_prev": lluvia_7d_prev.values})

    if df_all.empty:
        st.error("Fusión vacía (ni histórico válido ni API).")
        st.stop()

    input_df_raw = df_all.copy()
    src = ["API"]
    if not df_hist_trim.empty:
        src.append(f"{hist_source_desc} ({df_hist_trim['Fecha'].min().date()} → {df_hist_trim['Fecha'].max().date()})")
    else:
        src.append(hist_source_desc)
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

# ================= Ejecutar modelo (intacto) =================
resultado = ejecutar_modelo(input_df, IW, bias_IW, LW, bias_out, umbral_usuario)

# Reemplazar Fecha por la del input original si está completa
fechas_excel = usar_fechas_de_input(input_df_raw, len(resultado))
if fechas_excel is not None:
    resultado["Fecha"] = fechas_excel

# ================= Rango 1-feb → 1-oct (ventana fija 2025) =================
# Mantengo helper 'reiniciar_feb_oct' y fuerzo la ventana fija con reindex:
pred_vis = reiniciar_feb_oct(resultado[["Fecha", "EMERREL (0-1)"]].copy(), umbral_ajustable=umbral_usuario)
pred_vis["Fecha"] = pd.to_datetime(pred_vis["Fecha"]).dt.normalize()

# Reindex a ventana completa fija (incluye fechas sin datos como NaN)
idx_fijo = pd.date_range(FECHA_INICIO_FIJA, FECHA_FIN_FIJA, freq="D")
pred_full = pd.DataFrame(index=idx_fijo).reset_index().rename(columns={"index": "Fecha"})
pred_full = pred_full.merge(pred_vis, on="Fecha", how="left")

# Sello y fuente (sin exponer URLs)
st.caption(f"Fuente de datos: {source_label}")
st.caption(f"Última actualización: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption(f"Umbral EMEAC usado: {umbral_usuario}" + (" (forzado desde código)" if usar_codigo else ""))

# ================= Clasificación (NO excluye días) + Lluvia 7d =================
try:
    pred_full = pred_full.merge(df_prec_lluvia, on="Fecha", how="left")
except Exception:
    if "lluvia_7d_prev" not in pred_full.columns:
        pred_full["lluvia_7d_prev"] = np.nan

def _nivel_base(v):
    if pd.isna(v):
        return np.nan
    return "Bajo" if v < 0.2 else ("Medio" if v < 0.4 else "Alto")

pred_full["Nivel_base"] = pred_full["EMERREL (0-1)"].apply(_nivel_base)
pred_full["gated_down"] = (
    APLICAR_REGLA_LLUVIA_7D
    & pred_full["Nivel_base"].isin(["Medio", "Alto"])
    & (pred_full["lluvia_7d_prev"].fillna(-1e9) < LLUVIA_CORTE_MM_7D)
)
pred_full["Nivel de EMERREL"] = np.where(pred_full["gated_down"], "Bajo", pred_full["Nivel_base"])

# ================= Series EMEAC (sobre TODA la serie, sin filtrar) =================
emerrel_series = pred_full["EMERREL (0-1)"].fillna(0.0).to_numpy()
cumsum_series = np.cumsum(emerrel_series)
emeac_min_pct = np.clip(cumsum_series / float(EMEAC_MAX) * 100.0, 0, 100)
emeac_max_pct = np.clip(cumsum_series / float(EMEAC_MIN) * 100.0, 0, 100)
emeac_ajust   = np.clip(cumsum_series / float(umbral_usuario) * 100.0, 0, 100)

# ================= Gráficos (Serie completa 1-feb → 1-oct 2025) =================
if PLOTLY_OK:
    color_map = {"Bajo": "green", "Medio": "yellow", "Alto": "red"}

    # --- Gráfico 1: EMERREL ---
    st.subheader("EMERGENCIA RELATIVA DIARIA - BORDENAVE (Serie completa 1-feb → 1-oct 2025)")
    fig1 = go.Figure()

    bar_colors = pred_full["Nivel_base"].map(color_map).fillna("lightgray").tolist()
    bar_opacity = np.where(pred_full["gated_down"], 0.45, 0.9).tolist()

    fig1.add_bar(
        x=pred_full["Fecha"],
        y=pred_full["EMERREL (0-1)"],
        marker=dict(color=bar_colors, opacity=bar_opacity),
        customdata=np.stack([
            pred_full["Nivel_base"].fillna("s/d"),
            pred_full["Nivel de EMERREL"].fillna("s/d"),
            pred_full["lluvia_7d_prev"].fillna(np.nan)
        ], axis=-1),
        hovertemplate=(
            "Fecha: %{x|%d-%b-%Y}"
            "<br>EMERREL: %{y:.3f}"
            "<br>Nivel base: %{customdata[0]}"
            "<br>Nivel final (regla): %{customdata[1]}"
            "<br>Lluvia 7d: %{customdata[2]:.1f} mm"
            "<extra></extra>"
        ),
        name="EMERREL (0-1)",
    )

    pred_full["EMERREL_MA5"] = pred_full["EMERREL (0-1)"].rolling(5, min_periods=1).mean()
    fig1.add_trace(go.Scatter(
        x=pred_full["Fecha"],
        y=pred_full["EMERREL_MA5"],
        mode="lines", name="Media móvil 5 días",
        hovertemplate="Fecha: %{x|%d-%b-%Y}<br>MA5: %{y:.3f}<extra></extra>"
    ))
    fig1.add_trace(go.Scatter(
        x=pred_full["Fecha"], y=pred_full["EMERREL_MA5"],
        mode="lines", line=dict(width=0),
        fill="tozeroy", fillcolor="rgba(135, 206, 250, 0.30)",
        name="Área MA5", hoverinfo="skip", showlegend=False
    ))

    y_low, y_med = 0.2, 0.4
    fig1.add_hline(y=y_low, line_dash="dot", annotation_text=f"Bajo (≤ {y_low:.2f})")
    fig1.add_hline(y=y_med, line_dash="dot", annotation_text=f"Medio (≤ {y_med:.2f})")
    fig1.update_xaxes(range=[FECHA_INICIO_FIJA, FECHA_FIN_FIJA])
    fig1.update_layout(xaxis_title="Fecha", yaxis_title="EMERREL (0-1)", hovermode="x unified", legend_title="Referencias", height=650)
    st.plotly_chart(fig1, use_container_width=True, theme="streamlit")

    # --- Gráfico 2: EMEAC ---
    st.subheader("EMERGENCIA ACUMULADA DIARIA - BORDENAVE (Serie completa 1-feb → 1-oct 2025)")
    st.markdown(f"**Umbrales:** Min={EMEAC_MIN} · Max={EMEAC_MAX} · Ajustable={umbral_usuario}")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=pred_full["Fecha"], y=emeac_min_pct, mode="lines", line=dict(width=0), name=f"Mínimo (umbral {EMEAC_MAX})",
                              hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Mínimo: %{y:.1f}%<extra></extra>"))
    fig2.add_trace(go.Scatter(x=pred_full["Fecha"], y=emeac_max_pct, mode="lines", line=dict(width=0), fill="tonexty", name=f"Máximo (umbral {EMEAC_MIN})",
                              hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Máximo: %{y:.1f}%<extra></extra>"))
    fig2.add_trace(go.Scatter(x=pred_full["Fecha"], y=emeac_ajust, mode="lines", name=f"Ajustable ({umbral_usuario})", line=dict(width=2.5),
                              hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Ajustable: %{y:.1f}%<extra></extra>"))
    for nivel in [25, 50, 75, 90]:
        fig2.add_hline(y=nivel, line_dash="dash", opacity=0.6, annotation_text=f"{nivel}%")
    fig2.update_xaxes(range=[FECHA_INICIO_FIJA, FECHA_FIN_FIJA])
    fig2.update_layout(xaxis_title="Fecha", yaxis_title="EMEAC (%)", hovermode="x unified", legend_title="Referencias", yaxis=dict(range=[0, 100]), height=600)
    st.plotly_chart(fig2, use_container_width=True, theme="streamlit")

else:
    # === Fallback Matplotlib ===
    color_map = {"Bajo": "green", "Medio": "yellow", "Alto": "red"}

    st.subheader("EMERGENCIA RELATIVA DIARIA - BORDENAVE (Serie completa 1-feb → 1-oct 2025)")
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ma5 = pred_full["EMERREL (0-1)"].rolling(5, min_periods=1).mean()
    ax1.fill_between(pred_full["Fecha"], 0, ma5, color="skyblue", alpha=0.3, zorder=0)
    bars_color = pred_full["Nivel_base"].map(color_map).fillna("lightgray")
    bars_alpha = np.where(pred_full["gated_down"], 0.45, 0.9)
    ax1.bar(pred_full["Fecha"], pred_full["EMERREL (0-1)"], color=bars_color, alpha=bars_alpha)
    ax1.plot(pred_full["Fecha"], ma5, linewidth=2.2, label="Media móvil 5 días")
    ax1.set_xlim(FECHA_INICIO_FIJA, FECHA_FIN_FIJA)
    ax1.set_ylabel("EMERREL (0-1)")
    ax1.legend(handles=[Patch(facecolor=color_map.get(k, "lightgray"), label=k) for k in ["Bajo","Medio","Alto"]], loc="upper right")
    ax1.grid(True); st.pyplot(fig1); plt.close(fig1)

    st.subheader("EMERGENCIA ACUMULADA DIARIA - BORDENAVE (Serie completa 1-feb → 1-oct 2025)")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(pred_full["Fecha"], emeac_ajust,   label=f"Ajustable ({umbral_usuario})", linewidth=2)
    ax2.plot(pred_full["Fecha"], emeac_min_pct, label=f"Mínimo (umbral {EMEAC_MAX})", linestyle="--", linewidth=2)
    ax2.plot(pred_full["Fecha"], emeac_max_pct, label=f"Máximo (umbral {EMEAC_MIN})", linestyle="--", linewidth=2)
    ax2.fill_between(pred_full["Fecha"], emeac_min_pct, emeac_max_pct, alpha=0.3, label="Área entre Mín y Máx")
    ax2.set_xlim(FECHA_INICIO_FIJA, FECHA_FIN_FIJA)
    ax2.set_ylabel("EMEAC (%)"); ax2.set_ylim(0, 105); ax2.legend(); ax2.grid(True)
    st.pyplot(fig2); plt.close(fig2)

# ================= Tabla — Serie completa (sin EMERREL/Aplicó regla/Nivel base) =================
pred_full["Día juliano"] = pred_full["Fecha"].dt.dayofyear
tabla_display = pd.DataFrame({
    "Fecha": pred_full["Fecha"],
    "Día juliano": pred_full["Día juliano"].astype(int),
    "Lluvia 7d (mm)": pred_full["lluvia_7d_prev"].round(1),
    "Nivel final": pred_full["Nivel de EMERREL"],
    "EMEAC (%)": emeac_ajust
})

st.subheader("Tabla de Resultados — Serie completa (1-feb → 1-oct 2025)")
st.dataframe(tabla_display, use_container_width=True)

csv_full = tabla_display.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Descargar tabla completa (1-feb → 1-oct 2025) en CSV",
    data=csv_full,
    file_name=f"tabla_completa_{pd.Timestamp.now().strftime('%Y-%m-%d_%H%M')}.csv",
    mime="text/csv"
)
