# -*- coding: utf-8 -*-
# meteobahia_api.py
from __future__ import annotations
import pandas as pd
import requests
import xml.etree.ElementTree as ET

def _get_attr_or_text(elem, attr="value"):
    if elem is None:
        return None
    v = elem.attrib.get(attr)
    return v if v is not None else (elem.text or None)

def parse_meteobahia_xml(xml_text: str) -> pd.DataFrame:
    """
    Parseo del XML de pronóstico (como https://meteobahia.com.ar/scripts/forecast/for-bd.xml).
    Extrae: Fecha, TMAX, TMIN, Prec → arma DataFrame estándar con Julian_days.
    """
    root = ET.fromstring(xml_text)
    days = root.findall(".//tabular/day")
    rows = []
    for d in days:
        fecha  = _get_attr_or_text(d.find("fecha"))
        tmax   = _get_attr_or_text(d.find("tmax"))
        tmin   = _get_attr_or_text(d.find("tmin"))
        precip = _get_attr_or_text(d.find("precip"))
        if not fecha:
            continue
        rows.append({
            "Fecha":  pd.to_datetime(fecha, errors="coerce"),
            "TMAX":   pd.to_numeric(tmax, errors="coerce"),
            "TMIN":   pd.to_numeric(tmin, errors="coerce"),
            "Prec":   pd.to_numeric(precip, errors="coerce"),
        })
    df = pd.DataFrame(rows).dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    return df[["Fecha","Julian_days","TMAX","TMIN","Prec"]]

def fetch_meteobahia_api_xml(
    url: str,
    *,
    token: str | None = None,
    timeout: int = 20,
    params: dict | None = None,
    use_browser_headers: bool = True,
) -> pd.DataFrame:
    """
    Descarga el XML y lo parsea. Por defecto envía headers 'de navegador' para evitar 403.
    """
    headers = {}
    if use_browser_headers:
        headers.update({
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/126.0.0.0 Safari/537.36"),
            "Referer": "https://meteobahia.com.ar/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "es-AR,es;q=0.9,en;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        })
    if token:
        headers["Authorization"] = f"Bearer {token}"

    r = requests.get(url, headers=headers, params=params or {}, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return parse_meteobahia_xml(r.text)
