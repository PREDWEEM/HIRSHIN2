
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Demo EMERREL", layout="wide")

# Simulación de datos (para ejemplo completo)
dates = pd.date_range(start="2025-02-01", end="2025-10-01", freq="D")
n = len(dates)
np.random.seed(42)
emerrel_values = np.random.rand(n) * 0.6  # valores entre 0.0 y 0.6
emeac_acum = np.clip(np.cumsum(emerrel_values) / 7 * 100, 0, 100)

# Crear DataFrame
df = pd.DataFrame({
    "Fecha": dates,
    "EMERREL (0-1)": emerrel_values,
    "EMEAC (%)": emeac_acum
})

# Clasificación y media móvil
df["EMERREL_MA5_rango"] = df["EMERREL (0-1)"].rolling(5, min_periods=1).mean()

def clasif(v):
    if v < 0.2:
        return "Bajo"
    elif v < 0.4:
        return "Medio"
    else:
        return "Alto"

df["Nivel de EMERREL"] = df["EMERREL (0-1)"].apply(clasif)
df["Día juliano"] = df["Fecha"].dt.dayofyear

# Íconos
nivel_icono = {
    "Bajo": "🟢 Bajo",
    "Medio": "🟠 Medio",
    "Alto": "🔴 Alto"
}
df["Nivel con ícono"] = df["Nivel de EMERREL"].map(nivel_icono)

# Mostrar tabla final
tabla = df[["Fecha", "Día juliano", "Nivel con ícono", "EMEAC (%)"]].rename(columns={
    "Nivel con ícono": "Nivel de EMERREL"
})

st.title("Tabla de Resultados (Demo EMERREL)")
st.dataframe(tabla, use_container_width=True)

# Descargar CSV
csv_data = tabla.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Descargar tabla en CSV",
    data=csv_data,
    file_name="tabla_rango_demo.csv",
    mime="text/csv"
)
