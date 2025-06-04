import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import holidays


def limpieza_reservaciones(reservaciones: pd.DataFrame) -> pd.DataFrame:
    """Preprocesa y limpia los datos de reservaciones.

    Args:
        iar_reservaciones: DataFrame crudo con las reservaciones.

    Returns:
        DataFrame limpio y preprocesado.
    """
    # Calcula el número total de personas (adultos + menores)
    #Completa los datos faltantes de adultos y menores usando las columnas auxiliares
    reservaciones['h_num_adu'] = (reservaciones['h_num_adu'].fillna(reservaciones['aa_h_num_adu']))
    reservaciones['h_num_men'] = (reservaciones['h_num_men'].fillna(reservaciones['aa_h_num_men']))
    reservaciones['h_num_per'] = reservaciones['h_num_adu']+reservaciones['h_num_men']

    # Convierte las columnas de fechas a formato datetime
    reservaciones['h_fec_lld_ok'] = pd.to_datetime(reservaciones['h_fec_lld_ok'])
    reservaciones['h_fec_sda_ok']  = pd.to_datetime(reservaciones['h_fec_sda_ok'])
    reservaciones['h_res_fec_ok']  = pd.to_datetime(reservaciones['h_res_fec_ok'])

    # Calcula el número de noches restando la fecha de llegada a la de salida
    delta = reservaciones['h_fec_sda_ok'] - reservaciones['h_fec_lld_ok']
    reservaciones['h_num_noc'] = delta.dt.days

    # Crea una máscara para identificar reservaciones con 0 noches
    mask_no_nights = reservaciones['h_num_noc'] == 0

    # Crea una máscara para conservar las reservaciones con 0 noches pero válidas:
    # que tengan estado 9 y al menos una persona
    mask_promote = (
        mask_no_nights &
        (reservaciones['ID_estatus_reservaciones'] == 9) &
        (reservaciones['h_num_per'] > 0)
    )

    # Asigna 1 noche a estas reservaciones válidas

    reservaciones.loc[mask_promote, 'h_num_noc'] = 1

    # Elimina las reservaciones con 0 noches que no son válidas
    to_drop = mask_no_nights & ~mask_promote
    reservaciones = reservaciones.loc[~to_drop].copy()

    # Filtra para mantener solo las reservaciones donde la fecha de llegada es anterior o igual a la de salida
    reservaciones = reservaciones[reservaciones['h_fec_lld_ok'] <= reservaciones['h_fec_sda_ok']].copy()

    # Reemplaza los valores negativos de tarifa total con 0
    filas_negativas = reservaciones['h_tfa_total'] < 0
    reservaciones.loc[filas_negativas, 'h_tfa_total'] = 0

    # Selecciona solo las columnas relevantes para el análisis o exportación
    reservaciones = reservaciones[['ID_Reserva','h_res_fec_ok','h_fec_lld_ok', 'h_fec_sda_ok','h_num_per','h_num_adu','h_num_men','h_num_noc','h_tot_hab','ID_Tipo_Habitacion','ID_canal','h_tfa_total','ID_Pais_Origen','ID_Paquete','ID_Agencia','ID_estatus_reservaciones']]
    return reservaciones


def preprocess_dashboard(resv: pd.DataFrame,canales:pd.DataFrame,estatus:pd.DataFrame,tipohabitaciones:pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for shuttles.

    Args:
        reservaciones: Raw data.
    Returns:
        Preprocessed data, with price converted to a float and d_check_complete,
        moon_clearance_complete converted to boolean.
    """


    canales.columns = ["ID_canal", "Canal_cve", "Canal_nombre"]

    estatus.columns = ["ID_estatus_reservaciones", "estatus_reservaciones_cve", "Estatus_nombre"]

    tipohabitaciones.columns = ["ID_Tipo_Habitacion", "Tipo_Habitacion_cve", "Tipo_Habitacion_nombre","Cupo", "Clasificación"]



    resv = resv.merge(canales, on="ID_canal", how="left")

    resv = resv.merge(estatus, on="ID_estatus_reservaciones", how="left")

    resv = resv.merge(tipohabitaciones, on="ID_Tipo_Habitacion", how="left")

    dashboard_reservations = resv.copy()
    return dashboard_reservations


def create_model_input_table_mean_boost_extra_features(resv: pd.DataFrame) -> pd.DataFrame:
    """Genera una tabla diaria para el LSTM con total, adultos y menores,
    aplicando un boost de media (pre vs post pandemia) a las tres series."""

    # 1. Filtrar reservas válidas y expandir por noches (lógica original)
    df = resv.loc[resv['h_num_noc'] > 0].copy()
    df = df.loc[df.index.repeat(df['h_num_noc'])]
    df['fecha_estancia'] = (
        df.groupby('ID_Reserva')
          .cumcount()
          .apply(pd.Timedelta, unit='D')
        + df['h_fec_lld_ok']
    )

    # 2. Agrupar por fecha:
    #    - total de personas (h_num_per) [original]
    #    - adultos (num_adultos)         [**nuevo**]
    #    - menores (num_menores)         [**nuevo**]
    serie_total   = df.groupby('fecha_estancia')['h_num_per'].sum()
    serie_adultos = df.groupby('fecha_estancia')['h_num_adu'].sum()   # Cambio: agregada
    serie_menores = df.groupby('fecha_estancia')['h_num_men'].sum()   # Cambio: agregada

    # 3. Definir máscaras de periodo pre y post pandemia (lógica original adaptada)
    mask_pre  = (serie_total.index >= '2019-02-13') & (serie_total.index < '2020-04-01')
    mask_post = (serie_total.index >= '2020-04-01') & (serie_total.index < '2021-01-01')

    # 4. Calcular ratio de boost:
    #    - total      [original adaptado]
    #    - adultos    [**nuevo**]
    #    - menores    [**nuevo**]
    ratio_total   = serie_total[mask_pre].mean()   - serie_total[mask_post].mean()
    ratio_adultos = serie_adultos[mask_pre].mean() - serie_adultos[mask_post].mean()  # Cambio: agregado
    ratio_menores = serie_menores[mask_pre].mean() - serie_menores[mask_post].mean()  # Cambio: agregado

    # 5. Aplicar boost aditivo en el periodo abril–diciembre 2020
    mask_boost = (serie_total.index >= '2020-04-01') & (serie_total.index < '2021-01-01')
    serie_total.loc[mask_boost]   += ratio_total
    serie_adultos.loc[mask_boost] += ratio_adultos  # Cambio: agregado
    serie_menores.loc[mask_boost] += ratio_menores  # Cambio: agregado

    # 6. Seleccionar rango final y reindexar:
    #    - el total ya está en rango [original]
    #    - adultos/menores se reindexan para alinear fechas [**nuevo**]
    inicio, fin = '2019-02-13', '2020-12-31'
    serie_total   = serie_total.loc[inicio:fin]
    serie_adultos = serie_adultos.reindex(serie_total.index, fill_value=0)
    serie_menores = serie_menores.reindex(serie_total.index, fill_value=0)

    # 7. Reconstruir rango diario completo y combinar las tres series
    full_range = pd.date_range(start=serie_total.index.min(),
                               end=serie_total.index.max(),
                               freq='D')
    out = pd.DataFrame(index=full_range)
    out['h_num_per']   = serie_total
    out['h_num_adu'] = serie_adultos  # Cambio: nueva columna
    out['h_num_men'] = serie_menores  # Cambio: nueva columna


    # 8) Interpolar las tres series
    out['h_num_per'] = out['h_num_per'].interpolate(method='linear')
    out['h_num_adu'] = out['h_num_adu'].interpolate(method='linear')
    out['h_num_men'] = out['h_num_men'].interpolate(method='linear')

    # 9) Convertir total y adultos a enteros
    out['h_num_per'] = out['h_num_per'].round().astype(int)
    out['h_num_adu'] = out['h_num_adu'].round().astype(int)

    # 10) Recalcular menores exactamente para que sumen
    out['h_num_men'] = out['h_num_per'] - out['h_num_adu']
 
    # 11) Reset index…
    model_input_table_boost = out.reset_index().rename(columns={'index':'fecha_estancia'})
    return model_input_table_boost

def prepare_lstm_data(resv: pd.DataFrame,
                      time_steps: int = 15,
                      test_size: float = 0.2) -> tuple:
    """
    1) Llama a create_model_input_table_mean_boost
    2) Crea features temporales y cíclicas
    3) Escala y genera secuencias
    4) Divide en train/test

    Retorna: X_train, X_test, y_train, y_test, scaler, df_feat, features
    """
    # 1) Boost + conteos diarios
    df = resv.sort_values('fecha_estancia').reset_index(drop=True)

    # 2) Features temporales y cíclicas
    df['dia_semana']     = df['fecha_estancia'].dt.dayofweek
    df['mes']            = df['fecha_estancia'].dt.month
    df['dia_año']        = df['fecha_estancia'].dt.dayofyear
    df['es_fin_semana']  = (df['dia_semana'] >= 5).astype(int)
    df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
    df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
    df['mes_sin']        = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos']        = np.cos(2 * np.pi * df['mes'] / 12)

    #2.1) Etiquetar dias festivos
    years = range(df['fecha_estancia'].dt.year.min(), df['fecha_estancia'].dt.year.max() + 2)
    mex_holidays = holidays.Mexico(years=years)
    df['is_holiday'] = df['fecha_estancia'].isin(mex_holidays).astype(int)

    # 3) Selección de features (incluye ahora adultos y menores)
    features = [
        'h_num_per','h_num_adu','h_num_men',
        'dia_semana_sin','dia_semana_cos',
        'mes_sin','mes_cos',
        'dia_año','es_fin_semana', 'is_holiday'
    ]

    # 4) Escalado
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])

    # 5) Crear secuencias deslizantes
    def create_sequences(arr, ts):
        X, y = [], []
        for i in range(len(arr) - ts):
            X.append(arr[i:i+ts])
            y.append(arr[i+ts, 0])  # objetivo: h_num_per
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled, time_steps)

    # 6) Dividir en train/test (sin shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    return X_train, X_test, y_train, y_test, scaler, df
