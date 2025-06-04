import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import numpy as np
import holidays
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import requests
import json
import os
from azure.storage.blob import BlobServiceClient
from io import BytesIO
from dotenv import load_dotenv


load_dotenv()

url = "http://localhost:8000/predict"

headers = {
    "Content-Type": "application/json"
}


# Cargar datos
#df = pd.read_csv("dashboard_reservations.csv", parse_dates=["h_res_fec_okt", "h_fec_lld_okt", "h_fec_sda_okt"])

#df = pd.read_parquet("data/03_primary/dashboard_reservations.parquet")



CONN_STR = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
CONTAINER_NAME = "datos"

blob_service_client = BlobServiceClient.from_connection_string(CONN_STR)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

# Descargar modelo
blob_client = container_client.get_blob_client("dashboard_reservations.parquet")

blob_data = blob_client.download_blob().readall()

df = pd.read_parquet(BytesIO(blob_data))
if df.empty:
    raise ValueError("El DataFrame está vacío. Verifica el archivo en Azure Blob Storage.")

# Convertir columnas a categóricas
cat_cols = ['ID_Tipo_Habitacion', 'ID_canal', 'ID_Pais_Origen', 'ID_Paquete', 'ID_Agencia', 'ID_estatus_reservaciones']
for col in cat_cols:
    df[col] = df[col].astype(str)

# Configuración general
st.set_page_config(
    page_title="ANÁLISIS DE RESERVAS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

    #Navegación de páginas
    

st.sidebar.image("tca2.png", width=250)

with st.sidebar:
    page = option_menu(
        menu_title="Menú Principal",  # Título visible
        options=["Dashboard Ejecutivo", "Análisis por Canal", "Dashboard Predicciones", "Datos"],
        icons=["bar-chart-line-fill", "graph-up","lightbulb-fill", "folder2-open"],  # Bootstrap icons
        menu_icon="tv-fill",  # ícono del encabezado
        default_index=0,
        styles={
            "container": {
                "background-color": "#111827",  # fondo oscuro sobrio
                "padding": "10px",
                "border-radius": "10px"
            },
            "icon": {
                "color": "white",
                "font-size": "18px"
            },
            "nav-link": {
                "color": "#e5e7eb",  # gris claro
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#374151"  # hover gris oscuro
            },
            "nav-link-selected": {
                "background-color": "#4B5563",  # gris elegante
                "color": "white",
                "font-weight": "bold"
            },
            "menu-title": {
                "color": "white",
                "font-size": "20px",
                "font-weight": "bold"
            }
        }
    )



if page == "Datos":
    st.title("Datos de Reservas")

    # Filtro de criterio de fecha
    criterio_fecha = st.selectbox(
        "📆 Selecciona el criterio de fecha:",
        options={
            "h_res_fec_ok": "Fecha de Reserva",
            "h_fec_lld_ok": "Fecha de Llegada",
            "h_fec_sda_ok": "Fecha de Salida"
        }.values()
    )

    # Mapeo
    map_columnas = {
        "Fecha de Reserva": "h_res_fec_ok",
        "Fecha de Llegada": "h_fec_lld_ok",
        "Fecha de Salida": "h_fec_sda_ok"
    }
    columna_fecha = map_columnas[criterio_fecha]

    # Rango combinado global
    min_fecha = min(df["h_res_fec_ok"].min(), df["h_fec_lld_ok"].min(), df["h_fec_sda_ok"].min())
    max_fecha = max(df["h_res_fec_ok"].max(), df["h_fec_lld_ok"].max(), df["h_fec_sda_ok"].max())

    rango_fecha = st.date_input(
        "📅 Rango de fechas:",
        value=(min_fecha, max_fecha),
        min_value=min_fecha,
        max_value=max_fecha
    )
    fecha_inicio, fecha_fin = rango_fecha

    # Filtrar el DataFrame
    df_datos = df[
        (df[columna_fecha] >= pd.to_datetime(fecha_inicio)) &
        (df[columna_fecha] <= pd.to_datetime(fecha_fin))
    ]

    st.dataframe(df_datos)

    st.download_button(
        label="Descargar CSV filtrado",
        data=df_datos.to_csv(index=False).encode('utf-8'),
        file_name='reservas_filtradas.csv',
        mime='text/csv'
    )

    st.markdown("###  Diccionario de columnas")

    with st.expander("🧭 Dimensiones"):
        st.markdown("""
        - **ID_Reserva**: Identificador único de la reserva  
        - **h_res_fec_ok**: Fecha en que se realizó la reserva  
        - **h_fec_lld_ok**: Fecha de llegada al hotel  
        - **h_fec_sda_ok**: Fecha de salida del hotel  
        - **ID_Tipo_Habitacion**: ID del tipo de habitación  
        - **Tipo_Habitacion_cve**: Clave interna del tipo de habitación  
        - **Tipo_Habitacion_nombre**: Nombre del tipo de habitación  
        - **ID_canal**: ID del canal de venta  
        - **Canal_cve**: Clave del canal  
        - **Canal_nombre**: Nombre del canal de reserva  
        - **ID_Pais_Origen**: ID del país de origen del huésped  
        - **ID_Paquete**: ID del paquete contratado (si aplica)  
        - **ID_Agencia**: ID de la agencia de viajes (si aplica)  
        - **ID_estatus_reservaciones**: ID del estatus de la reserva  
        - **estatus_reservaciones_cve**: Clave del estatus  
        - **Estatus_nombre**: Nombre del estatus de la reserva  
        - **Clasificación**: Clasificación comercial de la habitación  
        """)

    with st.expander("📏 Medidas"):
        st.markdown("""
        - **h_num_per**: Total de personas en la reserva  
        - **h_num_adu**: Número de adultos  
        - **h_num_men**: Número de menores  
        - **h_num_noc**: Número de noches  
        - **h_tot_hab**: Total de habitaciones reservadas  
        - **h_tfa_total**: Monto total pagado por la reserva  
        - **Cupo**: Capacidad máxima de personas por habitación  
        """)

    st.stop()

    
elif page == "Dashboard Predicciones":

    # Configuración de Streamlit
    st.title("Predicción de Ocupación Hotelera")
    horizon = st.number_input("Número de días a predecir", min_value=1, max_value=730, value=30, step=1)
    tarifa_promedio = st.number_input("Tarifa promedio por persona (MXN):", value=1200.0, min_value=0.0)

    # Constantes
    time_steps = 15
    features = [
        'h_num_per', 'h_num_adu', 'h_num_men',
        'dia_semana_sin', 'dia_semana_cos',
        'mes_sin', 'mes_cos',
        'dia_año', 'es_fin_semana', 'is_holiday'
    ]

    # Cargar datos y modelo
    blob_feat = container_client.get_blob_client("lstm_df_feat.parquet")
    feat_data = blob_feat.download_blob().readall()
    df_feat = pd.read_parquet(BytesIO(feat_data))
    #df_feat = pd.read_parquet('data/04_feature/lstm_df_feat.parquet')
    #model = load_model("data/06_models/modelo.keras")

    # Escalar datos
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_feat[features])
    min_v, max_v = scaler.data_min_[0], scaler.data_max_[0]
    last_date = df_feat['fecha_estancia'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')

    # Reconstrucción de features para histórico + futuro
    all_dates = pd.DatetimeIndex(df_feat['fecha_estancia']).union(future_dates)
    full_df = pd.DataFrame(index=all_dates)
    full_df['dia_semana'] = all_dates.dayofweek
    full_df['mes'] = all_dates.month
    full_df['dia_año'] = all_dates.dayofyear
    full_df['es_fin_semana'] = (full_df['dia_semana'] >= 5).astype(int)
    full_df['dia_semana_sin'] = np.sin(2 * np.pi * full_df['dia_semana'] / 7)
    full_df['dia_semana_cos'] = np.cos(2 * np.pi * full_df['dia_semana'] / 7)
    full_df['mes_sin'] = np.sin(2 * np.pi * full_df['mes'] / 12)
    full_df['mes_cos'] = np.cos(2 * np.pi * full_df['mes'] / 12)
    years = range(full_df.index.year.min(), full_df.index.year.max() + 2)
    mex_holidays = holidays.Mexico(years=years)
    full_df['is_holiday'] = full_df.index.isin(mex_holidays).astype(int)

    # Inyectar histórico
    full_df['h_num_per'] = np.nan
    full_df['h_num_adu'] = np.nan
    full_df['h_num_men'] = np.nan
    hist_idx = df_feat['fecha_estancia']
    full_df.loc[hist_idx, 'h_num_per'] = df_feat['h_num_per'].values
    full_df.loc[hist_idx, 'h_num_adu'] = df_feat['h_num_adu'].values
    full_df.loc[hist_idx, 'h_num_men'] = df_feat['h_num_men'].values

    # Proporción media de adultos
    ratios_adu = df_feat['h_num_adu'] / df_feat['h_num_per']
    mean_ratios_adu = ratios_adu.groupby(df_feat['fecha_estancia'].dt.dayofweek).mean()

    # Escalar
    scaled_full = scaler.transform(full_df[features].fillna(0))

    # Preparar predicción
    window = scaled_full[:len(df_feat)].copy()
    _, _, input_dim = (None, 15, 10)
    preds_scaled = []
    min_adu, max_adu = scaler.data_min_[features.index('h_num_adu')], scaler.data_max_[features.index('h_num_adu')]
    min_men, max_men = scaler.data_min_[features.index('h_num_men')], scaler.data_max_[features.index('h_num_men')]

    # Bucle autoregresivo
    if st.button("Iniciar predicción", key="start_prediction"):
        for i, f_date in enumerate(future_dates):
            seq = window[-time_steps:].reshape(1, time_steps, input_dim)
            #making seq for api call to dict
            payload = {
                "data": seq.tolist()
            }
            response = requests.post(url, data=json.dumps(payload), headers=headers)

            next_s = response.json()['prediction'][0][0]
            preds_scaled.append(next_s)

            row = scaled_full[len(df_feat) + i].copy()
            row[features.index('h_num_per')] = next_s

            dow = f_date.dayofweek
            adu_pred = next_s * (max_v - min_v) + min_v
            adu_pred *= mean_ratios_adu.loc[dow]
            men_pred = (next_s * (max_v - min_v) + min_v) - adu_pred

            adu_scaled = (adu_pred - min_adu) / (max_adu - min_adu)
            men_scaled = (men_pred - min_men) / (max_men - min_men)
            row[features.index('h_num_adu')] = adu_scaled
            row[features.index('h_num_men')] = men_scaled

            window = np.vstack([window, row])

        # Desescalar predicción
        future_inv = np.array(preds_scaled) * (max_v - min_v) + min_v
        df_pred = pd.DataFrame({
            'Fecha': future_dates,
            'Número de personas': np.round(future_inv).astype(int),
            'Ingreso estimado (MXN)': np.round(future_inv * tarifa_promedio, 2)
        })

        # Gráfica interactiva
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_feat['fecha_estancia'], y=df_feat['h_num_per'],
            mode='lines', name='Histórico',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=df_pred['Fecha'], y=df_pred['Número de personas'],
            mode='lines', name='Predicción',
            line=dict(color='red'),
            hovertemplate='Fecha: %{x}<br>Personas: %{y}<extra></extra>'
        ))
        fig.update_layout(title='Ocupación de hotel: histórico y predicción',
                        xaxis_title='Fecha',
                        yaxis_title='Número de personas',
                        hovermode='x unified')

        st.plotly_chart(fig, use_container_width=True)

        # Mostrar tabla de resultados
        st.subheader("Predicciones detalladas")
        st.dataframe(df_pred.style.format({
            'Número de personas': '{:,.0f}',
            'Ingreso estimado (MXN)': '${:,.2f}'
        }), use_container_width=True)

    st.stop()
    
elif page == "Análisis por Canal":
    st.title(" Análisis por Canal de Reserva")

    st.sidebar.markdown("## 📅 Filtro por Fecha de Reserva")

    min_reserva = df["h_res_fec_ok"].min()
    max_reserva = df["h_res_fec_ok"].max()
    # Selección del rango
    rango_reserva = st.sidebar.date_input(
        "Selecciona rango de fechas:",
        value=(min_reserva, max_reserva),
        min_value=min_reserva,
        max_value=max_reserva
    )

    # Asegurar que siempre haya un rango válido
    if isinstance(rango_reserva, tuple) and len(rango_reserva) == 2:
        fecha_ini_reserva, fecha_fin_reserva = rango_reserva
    else:
        fecha_ini_reserva, fecha_fin_reserva = min_reserva,max_reserva
        
    df_canal = df.copy()

    colc1, colc2 = st.columns(2)

    # Gráfico 1: Pie chart de reservas por canal
    with colc1:
        ##st.subheader(" Distribución de Reservas por Canal")
        reservas_por_canal = (
            df_canal.groupby("Canal_nombre")["ID_Reserva"]
            .count()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig1 = px.pie(
            reservas_por_canal,
            names="Canal_nombre",
            values="ID_Reserva",
            title=" Reservas por Canal",
            hole=0.4
        )
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1, use_container_width=True)

    # Gráfico 2: Barras de ingresos por canal
    with colc2:
        ##st.subheader(" Ingresos por Canal")
        ingresos_por_canal = (
            df_canal.groupby("Canal_nombre")["h_tfa_total"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig2 = px.bar(
            ingresos_por_canal,
            x="Canal_nombre",
            y="h_tfa_total",
            labels={"h_tfa_total": "Ingreso Total (MXN)", "Canal_nombre": "Canal"},
            title=" Ingresos Totales por Canal",
            text_auto=True
        )
        fig2.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig2, use_container_width=True)


    
    
    ##st.markdown("###  Histórico de Reservas por Canal (Top 5)")

    # Agrupar por canal y contar reservas totales
    top_canales = (
        df_canal.groupby("Canal_nombre")["ID_Reserva"]
        .count()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )

    # Filtrar solo esos canales
    df_top_canales = df_canal[df_canal["Canal_nombre"].isin(top_canales)]

    # Agrupar por fecha y canal
    historico = (
        df_top_canales.groupby(["h_res_fec_ok", "Canal_nombre"])["ID_Reserva"]
        .count()
        .reset_index()
        .rename(columns={"ID_Reserva": "Reservas"})
    )

    # Gráfico de líneas
    fig_lineas = px.line(
        historico,
        x="h_res_fec_ok",
        y="Reservas",
        color="Canal_nombre",
        labels={
            "h_res_fec_ok": "Fecha de Reserva",
            "Reservas": "Número de Reservas",
            "Canal_nombre": "Canal"
        },
        title=" Tendencia Histórica de Reservas - Top 5 Canales"
    )

    fig_lineas.update_layout(
        hovermode="x unified",
        legend_title_text="Canal",
        xaxis_title="Fecha",
        yaxis_title="Reservas",
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig_lineas, use_container_width=True)
    
    
        # Tabla detallada
    st.subheader(" Detalles por Canal")
    tabla_canal = df_canal.groupby("Canal_nombre").agg(
        Reservas=("ID_Reserva", "count"),
        Ingreso_Total_MXN=("h_tfa_total", "sum"),
        Noches_Promedio=("h_num_noc", "mean"),
        Personas_Promedio=("h_num_per", "mean")
    ).sort_values("Ingreso_Total_MXN", ascending=False).reset_index()

    # Mostrar tabla con formato amigable
    st.dataframe(
        tabla_canal.style.format({
            "Ingreso_Total_MXN": "${:,.2f}",
            "Noches_Promedio": "{:.1f}",
            "Personas_Promedio": "{:.1f}"
        }),
        use_container_width=True
    )

    st.stop()


st.sidebar.markdown("---")

st.sidebar.markdown("## 📅 Filtro por Fecha de Reserva")

min_reserva = df["h_res_fec_ok"].min()
max_reserva = df["h_res_fec_ok"].max()

# Selección del rango
rango_reserva = st.sidebar.date_input(
    "Selecciona rango de fechas:",
    value=(min_reserva, max_reserva),
    min_value=min_reserva,
    max_value=max_reserva
)

# Asegurar que siempre haya un rango válido
if isinstance(rango_reserva, tuple) and len(rango_reserva) == 2:
    fecha_ini_reserva, fecha_fin_reserva = rango_reserva
else:
    fecha_ini_reserva, fecha_fin_reserva = min_reserva, max_reserva


st.sidebar.markdown("---")
st.sidebar.markdown("##  🔎 Filtros de Exploración")

with st.sidebar.expander(" Canal de Reserva", expanded=False):
    canales = sorted(df["Canal_nombre"].dropna().unique())
    canal = st.multiselect(
        label="Selecciona canal(es)",
        options=canales,
        default=None,
        placeholder="Buscar canal..."
    )
    if not canal:
        canal = canales  # Si no se selecciona nada, usa todos

with st.sidebar.expander(" Estatus de Reserva", expanded=False):
    estatuses = sorted(df["Estatus_nombre"].dropna().unique())
    estatus = st.multiselect(
        label="Selecciona estatus",
        options=estatuses,
        default=None,
        placeholder="Buscar estatus..."
    )
    if not estatus:
        estatus = estatuses

with st.sidebar.expander(" Tipo de Habitación", expanded=False):
    tipos_hab = sorted(df["Tipo_Habitacion_nombre"].dropna().unique())
    habitacion = st.multiselect(
        label="Selecciona tipo(s)",
        options=tipos_hab,
        default=None,
        placeholder="Buscar tipo de habitación..."
    )
    if not habitacion:
        habitacion = tipos_hab




df_filtrado = df[
    (df["Canal_nombre"].isin(canal)) &
    (df["Estatus_nombre"].isin(estatus)) &
    (df["Tipo_Habitacion_nombre"].isin(habitacion)) &
    (df["h_res_fec_ok"] >= pd.to_datetime(fecha_ini_reserva)) &
    (df["h_res_fec_ok"] <= pd.to_datetime(fecha_fin_reserva))
]


# Título
st.markdown("## Análisis de Desempeño de Reservas")
st.markdown("### Resumen Ejecutivo")

# KPIs
col1, col2, col3, col4, col5 = st.columns(5)
# Formatear número de reservas únicas

res_uniques = df_filtrado["ID_Reserva"].nunique()

if res_uniques >= 1_000_000:
    formatted_reservas = f"{res_uniques / 1_000_000:.1f}M"
elif res_uniques >= 1_000:
    formatted_reservas = f"{res_uniques / 1_000:.1f}K"
else:
    formatted_reservas = str(res_uniques)

col1.metric("**Reservas únicas**", formatted_reservas)


# Formatear monto total a K o M
total_ingresos = df_filtrado["h_tfa_total"].sum()

if total_ingresos >= 1_000_000:
    formatted_ingresos = f"${total_ingresos/1_000_000:.1f}M"
elif total_ingresos >= 1_000:
    formatted_ingresos = f"${total_ingresos/1_000:.1f}K"
else:
    formatted_ingresos = f"${total_ingresos:,.0f}"

col2.metric("**Total ingresos**", formatted_ingresos)

prom_noches = df_filtrado["h_num_noc"].mean()
col3.metric("**Duración Promedio**", f"{prom_noches:.1f} ")

prom_personas = round(df_filtrado["h_num_per"].mean())
col4.metric("**Personas Promedio**", f"{prom_personas} ")


ocupacion_prom = df_filtrado["h_num_per"].sum() / df_filtrado["h_tot_hab"].sum()

if ocupacion_prom > 1:
    col5.metric("**Ocupación/Habitación**", f"{ocupacion_prom:.2f}")
else:
    porcentaje = ocupacion_prom * 100
    col5.metric("**Tasa de ocupación por habitación**", f"{porcentaje:.1f}%")

# Mostrar porcentaje de cancelaciones usando columna Estatus_nombre
canceladas = df_filtrado[df_filtrado["Estatus_nombre"] == "RESERVACION CANCELADA"].shape[0]
pct_cancel = (canceladas / df_filtrado.shape[0] * 100) if df_filtrado.shape[0] > 0 else 0
st.info(f"🔔 **Porcentaje de reservas canceladas:** {pct_cancel:.1f}%")


#--GRÁFICAS--

# --- BLOQUE 1: Reservas e ingresos por canal ---


# 1. Reservas por fecha
resxfecha = df_filtrado.groupby("h_res_fec_ok")["ID_Reserva"].count().reset_index()
fig_fecha = px.line(resxfecha, x="h_res_fec_ok", y="ID_Reserva", title=" Reservas por Fecha")
st.plotly_chart(fig_fecha, use_container_width=True)



# --- BLOQUE 2: Tipo de habitación (duración e ingresos) ---
col8, col9 = st.columns(2)

# 3. Noches promedio por tipo de habitación (Top 20)
noches_hab = (
    df_filtrado.groupby("Tipo_Habitacion_nombre")["h_num_noc"]
    .mean()
    .sort_values(ascending=False)
    .head(20)
    .reset_index()
)
fig_noches = px.bar(
    noches_hab,
    x="Tipo_Habitacion_nombre",
    y="h_num_noc",
    title=" Noches Promedio por Tipo de Habitación (Top 20)",
    labels={"h_num_noc": "Noches Promedio", "Tipo_Habitacion_nombre": "Tipo de Habitación"},
    text_auto=True
)
fig_noches.update_layout(xaxis={'categoryorder': 'total descending'})
col8.plotly_chart(fig_noches, use_container_width=True)

# 4. Total pagado por tipo de habitación
hab_ingresos = df_filtrado.groupby("Tipo_Habitacion_nombre")["h_tfa_total"].sum().reset_index()
hab_ingresos = hab_ingresos.sort_values("h_tfa_total", ascending=False)
fig_habitacion = px.bar(
    hab_ingresos,
    x="Tipo_Habitacion_nombre",
    y="h_tfa_total",
    title=" Ingresos por Tipo de Habitación",
    labels={"h_tfa_total": "Monto Total Pagado", "Tipo_Habitacion_nombre": "Tipo de Habitación"},
    text_auto=True
)
fig_habitacion.update_layout(xaxis={'categoryorder': 'total descending'})
col9.plotly_chart(fig_habitacion, use_container_width=True)


# --- BLOQUE 3: Distribución por duración y tipo de huésped ---
col10, col11 = st.columns(2)

# 5. Distribución de noches reservadas
df_copia = df_filtrado.copy()
df_copia["h_num_noc_group"] = df_copia["h_num_noc"].apply(lambda x: str(int(x)) if x <= 10 else "11")
frecuencias = df_copia["h_num_noc_group"].value_counts().reset_index()
frecuencias.columns = ['Numero de noches', 'Frecuencia']
orden = [str(i) for i in range(1, 11)] + ["11"]
frecuencias["Numero de noches"] = pd.Categorical(frecuencias["Numero de noches"], categories=orden, ordered=True)
frecuencias = frecuencias.sort_values("Numero de noches")
fig_frec_noches = px.bar(
    frecuencias,
    x='Numero de noches',
    y='Frecuencia',
    title=' Distribución de Noches por Reserva (0–10, agrupado "11")',
    text_auto=True
)
col10.plotly_chart(fig_frec_noches, use_container_width=True)

# 6. Composición adultos vs menores
total = df_filtrado[["h_num_adu", "h_num_men"]].sum().reset_index()
total.columns = ["Grupo", "Cantidad"]
fig_composicion = px.bar(
    total,
    x="Grupo",
    y="Cantidad",
    title=" Composición de Huéspedes: Adultos vs Menores",
    labels={"Cantidad": "Número de Personas"},
    text_auto=True,
    color="Grupo"
)
col11.plotly_chart(fig_composicion, use_container_width=True)


# --- BLOQUE FINAL: Distribución por país ---

# Mapeo de ID a nombres
mapa_paises = {
    157: "México",
    0: "Sin definir",
    38: "Canadá",
    232: "Estados Unidos"
}

# Agrupar por ID y contar reservas
pais = df_filtrado.groupby("ID_Pais_Origen")["ID_Reserva"].count().reset_index()

# Asegurar que los IDs sean enteros (por si vienen como string)
pais["ID_Pais_Origen"] = pais["ID_Pais_Origen"].astype(int)

# Aplicar mapeo
pais["País"] = pais["ID_Pais_Origen"].map(mapa_paises)

# Filtrar solo países definidos
pais = pais[pais["País"].notna()]

# Graficar
fig_pais = px.pie(
    pais,
    names="País",
    values="ID_Reserva",
    title=" Reservas por País de Origen",
    hole=0.4
)

fig_pais.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig_pais, use_container_width=True)
