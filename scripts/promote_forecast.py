import os
import io
import sys
import csv
import json
import time
import math
import base64
import zipfile
import logging
from datetime import datetime
from dateutil import tz
from pathlib import Path

import pandas as pd
import requests
from xml.etree import ElementTree as ET

# ==============================
# Config desde variables de entorno
# ==============================
API_URL   = os.getenv("API_URL", "https://meteobahia.com.ar/scripts/forecast/for-bd.xml").strip()
PRON_DIAS = int(os.getenv("PRON_DIAS", "8"))
TZ_NAME   = os.getenv("TIMEZONE", "America/Argentina/Buenos_Aires")
GH_PATH   = os.getenv("GH_PATH", "data/historico.csv").strip()
API_TOKEN = os.getenv("API_TOKEN", "").strip()

# ==============================
# Utilidades
# ==============================
def now_local(tz_name: str) -> pd.Timestamp:
    return pd.Timestamp.now(tz=tz_name).normalize()

def _normalize_hist_like(df: pd.DataFrame, year_hint: int) -> pd.DataFrame:
    """
    Retorna columnas canon: Fecha, Julian_days, TMAX, TMIN, Prec
    Acepta CSV/XLSX previos con nombres variables (tmax/tmin/prec/julian/fecha).
    """
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
        (["tmax","t_max","tx","tmax(°c)"], "TMAX"),
        (["tmin","t_min","tn","tmin(°c)"], "TMIN"),
        (["prec","ppt","precip","lluvia","prcp","mm"], "Prec"),
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
        base = pd.Timestamp(int(year_hint), 1, 1)
        out["Fecha"] = out["Julian_days"].astype(float).apply(
            lambda d: base + pd.Timedelta(days=int(d) - 1)
        )

    if "Julian_days" not in out.columns and "Fecha" in out.columns:
        out["Julian_days"] = pd.to_datetime(out["Fecha"]).dt.dayofyear

    req = {"Fecha","Julian_days","TMAX","TMIN","Prec"}
    # Si faltan, devolvemos lo que haya para ver errores aguas arriba
    if not req.issubset(set(out.columns)):
        # intentamos al menos quedarnos con columnas compatibles
        cols = [c for c in ["Fecha","Julian_days","TMAX","TMIN","Prec"] if c in out.columns]
        out = out[cols]
        # completar si se puede
        if "Fecha" in out.columns and "Julian_days" not in out.columns:
            out["Julian_days"] = pd.to_datetime(out["Fecha"]).dt.dayofyear
        if "Julian_days" in out.columns and "Fecha" not in out.columns:
            base = pd.Timestamp(int(year_hint), 1, 1)
            out["Fecha"] = out["Julian_days"].astype(float).apply(
                lambda d: base + pd.Timedelta(days=int(d) - 1)
            )

    out = out.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    out["Julian_days"] = pd.to_datetime(out["Fecha"]).dt.dayofyear
    for c in ["TMAX","TMIN","Prec"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out[["Fecha","Julian_days","TMAX","TMIN","Prec"]]

def _read_history(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    try:
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p)
        elif p.suffix.lower() in (".xlsx", ".xls"):
            return pd.read_excel(p)
        else:
            # por compatibilidad, intentamos CSV
            return pd.read_csv(p)
    except Exception as e:
        print(f"[WARN] No pude leer histórico {path}: {e}")
        return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])

def _write_history(path: str, df: pd.DataFrame) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".csv":
        df.to_csv(p, index=False, date_format="%Y-%m-%d")
    elif p.suffix.lower() in (".xlsx", ".xls"):
        # mantiene tipos; la app igual puede leer CSV si preferís
        with pd.ExcelWriter(p, engine="openpyxl", mode="w") as xw:
            df.to_excel(xw, sheet_name="historico", index=False)
    else:
        df.to_csv(p, index=False, date_format="%Y-%m-%d")

# ==============================
# Lectura API (preferir módulo local; si no, fallback)
# ==============================
def fetch_api_df(api_url: str, token: str | None, use_browser_headers: bool = True) -> pd.DataFrame:
    """
    Intenta usar meteobahia_api.fetch_meteobahia_api_xml si existe en el repo.
    Si no, hace un parse "genérico" del XML buscando campos típicos.
    Debe retornar columnas: Fecha, TMAX, TMIN, Prec
    """
    # 1) Intento con módulo local
    try:
        from meteobahia_api import fetch_meteobahia_api_xml  # type: ignore
        df = fetch_meteobahia_api_xml(api_url, token=token or None, use_browser_headers=use_browser_headers)
        # Asegurar tipos
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
        for c in ["TMAX","TMIN","Prec"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    except Exception as e:
        print(f"[INFO] No se pudo usar meteobahia_api.fetch_meteobahia_api_xml ({e}). Intentando parser genérico…")

    # 2) Fallback “genérico”
    headers = {"User-Agent": "Mozilla/5.0"} if use_browser_headers else {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(api_url, headers=headers, timeout=30)
    r.raise_for_status()
    xml = ET.fromstring(r.content)

    # Heurística básica: buscar nodos de día que tengan fecha y valores
    rows = []
    # Intenta varios nombres típicos
    day_tags = ["day", "dia", "forecastday", "d", "item"]
    keys_date = ["date", "fecha", "day", "d"]
    keys_tmax  = ["tmax", "tx", "tempmax", "temp_max"]
    keys_tmin  = ["tmin", "tn", "tempmin", "temp_min"]
    keys_prec  = ["prec", "prcp", "pp", "lluvia", "rain"]

    def find_in(node, names):
        # busca atributo o subtag por nombres alternativos
        for k in names:
            # atributo
            if k in node.attrib:
                return node.attrib[k]
            # subtag
            sub = node.find(k)
            if sub is not None and (sub.text is not None):
                return sub.text
        # fallback: busca cualquier subtag que contenga el nombre
        for child in list(node):
            tagl = child.tag.lower()
            for k in names:
                if k in tagl and child.text is not None:
                    return child.text
        return None

    for dt in day_tags:
        for node in xml.findall(f".//{dt}"):
            s_date = find_in(node, keys_date)
            if not s_date:
                continue
            try:
                fecha = pd.to_datetime(s_date, errors="coerce")
            except Exception:
                fecha = pd.NaT
            if pd.isna(fecha):
                continue
            tmax = find_in(node, keys_tmax)
            tmin = find_in(node, keys_tmin)
            prec = find_in(node, keys_prec)
            try:
                tmax = float(str(tmax).replace(",", ".")) if tmax is not None else None
                tmin = float(str(tmin).replace(",", ".")) if tmin is not None else None
                prec = float(str(prec).replace(",", ".")) if prec is not None else None
            except Exception:
                tmax = None; tmin = None; prec = None
            rows.append({"Fecha": fecha, "TMAX": tmax, "TMIN": tmin, "Prec": prec})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Parser genérico no encontró días en el XML")
    df = df.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    for c in ["TMAX","TMIN","Prec"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ==============================
# Promoción al histórico
# ==============================
def promote_forecast_into_history(df_hist: pd.DataFrame, df_api: pd.DataFrame, tz_name: str) -> pd.DataFrame:
    """
    Concatena histórico + días de API con Fecha <= hoy_local. Elimina duplicados por Fecha (conserva la PRIMERA aparición).
    """
    if df_api is None or df_api.empty:
        return _normalize_hist_like(df_hist, year_hint=pd.Timestamp.now().year)

    df_api = df_api.copy()
    df_api["Fecha"] = pd.to_datetime(df_api["Fecha"], errors="coerce")
    df_api = df_api.dropna(subset=["Fecha"])
    df_api["Fecha"] = df_api["Fecha"].dt.tz_localize(None)

    # “Hoy” en BA
    hoy_local = now_local(tz_name)
    # Quedarnos solo con vencidos (<= hoy local)
    vencidos = df_api.loc[df_api["Fecha"] <= hoy_local.tz_localize(None)]
    if vencidos.empty:
        return _normalize_hist_like(df_hist, year_hint=hoy_local.year)

    # Normalizamos histórico y concatenamos
    year_hint = int(vencidos["Fecha"].min().year)
    hist_norm = _normalize_hist_like(df_hist, year_hint=year_hint)
    merged = pd.concat([hist_norm.sort_values("Fecha"), vencidos], ignore_index=True)
    merged = (
        merged.dropna(subset=["Fecha"])
              .sort_values(["Fecha"])
              .drop_duplicates(subset=["Fecha"], keep="first")  # mantiene el valor previo si existía
              .reset_index(drop=True)
    )
    merged["Julian_days"] = pd.to_datetime(merged["Fecha"]).dt.dayofyear
    for c in ["TMAX","TMIN","Prec"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
    return merged[["Fecha","Julian_days","TMAX","TMIN","Prec"]]

# ==============================
# Main
# ==============================
def main():
    print(f"[INFO] API_URL={API_URL}")
    print(f"[INFO] GH_PATH={GH_PATH}")
    print(f"[INFO] TIMEZONE={TZ_NAME}; PRON_DIAS={PRON_DIAS}")

    # 1) Leer API
    df_api = fetch_api_df(API_URL, token=API_TOKEN or None, use_browser_headers=True)
    if df_api is None or df_api.empty:
        print("[ERROR] API sin datos")
        sys.exit(1)

    # Orden cronológico y quedarnos con PRON_DIAS días únicos
    df_api["Fecha"] = pd.to_datetime(df_api["Fecha"], errors="coerce")
    df_api = df_api.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    dias_unicos = df_api["Fecha"].dt.normalize().unique()
    df_api = df_api[df_api["Fecha"].dt.normalize().isin(dias_unicos[:PRON_DIAS])].copy()

    # 2) Leer histórico actual
    df_hist = _read_history(GH_PATH)

    # 3) Promover vencidos
    df_new = promote_forecast_into_history(df_hist, df_api, tz_name=TZ_NAME)

    # 4) Guardar si hay cambios
    before = (len(df_hist), df_hist["Fecha"].max() if "Fecha" in df_hist.columns and not df_hist.empty else None)
    after  = (len(df_new),  df_new["Fecha"].max() if "Fecha" in df_new.columns  and not df_new.empty  else None)

    if df_new.empty:
        print("[WARN] No hay datos para escribir (df_new vacío).")
        sys.exit(0)

    # Comparamos por contenido
    same_shape = (set(df_hist.columns) == set(df_new.columns)) and (len(df_hist) == len(df_new))
    same_equal  = False
    if same_shape:
        try:
            same_equal = df_hist.sort_values("Fecha").reset_index(drop=True).equals(
                df_new.sort_values("Fecha").reset_index(drop=True)
            )
        except Exception:
            same_equal = False

    if same_equal:
        print("[INFO] Sin cambios respecto del histórico actual.")
        sys.exit(0)

    _write_history(GH_PATH, df_new)
    print(f"[OK] Histórico actualizado: filas {before[0]} → {after[0]} | max Fecha {before[1]} → {after[1]}")

if __name__ == "__main__":
    main()

