
# -*- coding: utf-8 -*-
"""
meteobahia.py
Utilidades de preparación/posproceso para MeteoBahía.
Incluye filtro 1-feb → 1-oct y reinicio de acumulados dentro del rango.
No modifica valores numéricos del modelo.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

UMBRAL_MIN = 9
UMBRAL_MAX = 17

def preparar_para_modelo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura nombres y tipos esperados por el modelo:
      - 'Julian_days' -> 'julian_days'
      - 'TMAX' -> 'tmax'
      - 'TMIN' -> 'Tmin'
      - 'Prec' -> 'prec'
    No cambia valores, solo normaliza nombres/tipos.
    """
    out = df.copy()
    out = out.rename(columns={
        "Julian_days": "julian_days",
        "TMAX": "tmax",
        "TMIN": "Tmin",
        "Prec": "prec"
    })
    for col in ["julian_days", "tmax", "Tmin", "prec"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "Fecha" in out.columns:
        out["Fecha"] = pd.to_datetime(out["Fecha"], errors="coerce")
    out = out.dropna(subset=["julian_days", "tmax", "Tmin", "prec"])
    return out

def usar_fechas_de_input(input_df: pd.DataFrame, n_filas: int):
    """
    Si el input trae 'Fecha' válida para todas las filas, devuélvela para
    sobreescribir la Fecha generada por el modelo; de lo contrario, devuelve None.
    """
    if "Fecha" in input_df.columns:
        f = pd.to_datetime(input_df["Fecha"], errors="coerce")
        if f.notna().sum() == n_filas:
            return f
    return None

def reiniciar_feb_oct(df_base: pd.DataFrame, umbral_ajustable: float) -> pd.DataFrame:
    """
    Recibe: DataFrame con columnas ['Fecha', 'EMERREL (0-1)'].
    Devuelve: subset 1-feb → 1-oct con acumulado reiniciado y curvas EMEAC (%)
              para umbrales Min (9), Max (17) y Ajustable (slider).
    No modifica valores del modelo; es posproceso/visualización.
    """
    if df_base.empty:
        return df_base.copy()

    fechas = pd.to_datetime(df_base["Fecha"])
    years = fechas.dt.year.unique()
    yr = int(years[0]) if len(years) == 1 else int(sorted(years)[-1])

    inicio = pd.Timestamp(year=yr, month=2, day=1)   # 1 de febrero
    fin    = pd.Timestamp(year=yr, month=10, day=1)  # 1 de octubre

    vis = df_base.loc[(fechas >= inicio) & (fechas <= fin)].copy()
    if vis.empty:
        return vis

    # Reinicio del acumulado en el rango
    vis["EMERREL acumulado (reiniciado)"] = vis["EMERREL (0-1)"].cumsum()

    # Curvas EMEAC (%) (sin tocar umbrales definidos)
    vis["EMEAC (%) - Min (rango)"]       = np.clip((vis["EMERREL acumulado (reiniciado)"] / float(UMBRAL_MIN)) * 100, 0, 100)
    vis["EMEAC (%) - Max (rango)"]       = np.clip((vis["EMERREL acumulado (reiniciado)"] / float(UMBRAL_MAX)) * 100, 0, 100)
    vis["EMEAC (%) - Ajustable (rango)"] = np.clip((vis["EMERREL acumulado (reiniciado)"] / float(umbral_ajustable)) * 100, 0, 100)

    # Media móvil 5 días dentro del rango
    vis["EMERREL_MA5_rango"] = vis["EMERREL (0-1)"].rolling(5, min_periods=1).mean()
    return vis
