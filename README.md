# Modelo EMERREL en Streamlit

Esta aplicación permite cargar un archivo con datos climáticos diarios y obtener predicciones de EMERREL y EMEAC, además de visualizar los niveles de riesgo.

## 📁 Requisitos

- Python 3.8+
- Streamlit

## 📦 Instalación

```bash
pip install -r requirements.txt
```

## 🚀 Ejecución local

```bash
streamlit run app.py
```

## 📝 Formato del archivo `input.xlsx`

El archivo debe tener las siguientes columnas:

| Julian_days | TMAX | TMIN | Prec |
|-------------|------|------|------|
| 1           | 32.5 | 20.1 | 0.0  |
| 2           | 34.0 | 21.3 | 0.0  |
| ...         | ...  | ...  | ...  |

- Julian_days: Día juliano (1–365)
- TMAX: Temperatura máxima en °C
- TMIN: Temperatura mínima en °C
- Prec: Precipitación en mm

## 📊 Salidas

- EMERREL clasificado como Bajo, Medio o Alto
- EMEAC acumulado en porcentaje
- Gráficos interactivos con umbrales visuales