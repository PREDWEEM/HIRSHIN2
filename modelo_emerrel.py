
import numpy as np
import pandas as pd
from modelo_restriccion_precipitacion_umbral16 import PracticalANNModel

def ejecutar_modelo(input_df, IW, bias_IW, LW, bias_out, umbral):
    class ModeloUmbral(PracticalANNModel):
        def __init__(self, IW, bias_IW, LW, bias_out, umbral):
            super().__init__(IW, bias_IW, LW, bias_out)
            self.umbral_emeac = umbral

    modelo = ModeloUmbral(IW, bias_IW, LW, bias_out, umbral)
    X_real = input_df[['julian_days', 'tmax', 'Tmin', 'prec']].values
    prec = input_df['prec'].values
    fechas = pd.date_range(start="2025-01-01", periods=len(input_df), freq='D')
    X_norm = modelo.normalize_input(X_real)
    emerrel_pred = np.array([modelo._predict_single(x) for x in X_norm])
    emerrel_desnorm = modelo.desnormalize_output(emerrel_pred).flatten()

    prec_acum_8 = np.convolve(prec, np.ones(8), mode='full')[:len(prec)]
    filtro_prec = prec_acum_8 >= 5
    emerrel_filtrado = emerrel_desnorm.copy()
    emerrel_filtrado[~filtro_prec] = 0

    emeac = np.cumsum(emerrel_filtrado) / modelo.umbral_emeac
    emeac_pct = np.clip(emeac * 100, 0, 100)

    return pd.DataFrame({
        "Fecha": fechas,
        "EMERREL (0-1)": emerrel_filtrado,
        "EMEAC (%)": emeac_pct
    })
