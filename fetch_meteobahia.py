# -*- coding: utf-8 -*-
"""
fetch_meteobahia.py
Carga robusta del CSV público de MeteoBahía desde GitHub Pages o Raw.
No modifica valores del modelo; solo trae datos y valida columnas.
"""

from __future__ import annotations
import pandas as pd

CSV_URL_PAGES = "https://GUILLE-bit.github.io/ANN/meteo_daily.csv"
CSV_URL_RAW   = "https://raw.githubusercontent.com/GUILLE-bit/ANN/gh-pages/meteo_daily.csv"

# Columnas mínimas esperadas (EMERREL proviene del modelo)
_REQUIRED_COLS = {"Fecha", "Julian_days", "TMAX", "TMIN", "Prec"}

def load_public_csv(parse_dates: bool = True) -> tuple[pd.DataFrame, str]:
    """
    Intenta leer el CSV primero desde Pages y luego desde Raw.
    Devuelve (df, url_de_éxito). Lanza RuntimeError si no puede.
    """
    last_err = None
    for url in (CSV_URL_PAGES, CSV_URL_RAW):
        try:
            df = pd.read_csv(url, parse_dates=["Fecha"] if parse_dates else None)

            # Normaliza/valida Fecha
            if "Fecha" not in df.columns:
                raise ValueError("Falta columna 'Fecha' en el CSV")
            df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
            if df["Fecha"].isna().any():
                raise ValueError("Fechas inválidas en el CSV")

            # Valida columnas mínimas
            missing = _REQUIRED_COLS - set(df.columns)
            if missing:
                raise ValueError(f"CSV sin columnas requeridas: {missing}")

            df = df.sort_values("Fecha").reset_index(drop=True)
            return df, url
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"No pude leer CSV (Pages/Raw). Último error: {last_err}")

def load_public_csv_between(start: str | pd.Timestamp,
                            end: str | pd.Timestamp) -> tuple[pd.DataFrame, str]:
    """
    Igual que load_public_csv pero recorta al rango [start, end].
    """
    df, url = load_public_csv(parse_dates=True)
    mask = (df["Fecha"] >= pd.to_datetime(start)) & (df["Fecha"] <= pd.to_datetime(end))
    return df.loc[mask].reset_index(drop=True), url

