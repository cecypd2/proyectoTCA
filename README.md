# 📈 Predicción de Reservaciones con Kedro y Despliegue en Azure

Este proyecto utiliza un pipeline de procesamiento y modelado con **Kedro**, y realiza el despliegue de una API y un dashboard usando **Docker** en un **Azure Container Instance**, con almacenamiento seguro en **Azure Blob Storage**. El objetivo es predecir el comportamiento de reservaciones hoteleras usando modelos de aprendizaje profundo y presentar los resultados en un entorno accesible y seguro.

---

## 🛠️ Requisitos

- Python 3.10
- Azure CLI
- Docker
- Kedro
- Cuenta de Azure con permisos para:
  - Storage Account
  - Container Registry
  - Container Instances

---

## 📁 Estructura del Proyecto

```
project/
│
├── data/
│   └── (archivos de datos y modelo entrenado)
│
├── conf/
│   ├── base/
│   └── local/
│
├── src/
│   ├── pipeline/           # Lógica Kedro
|
│── dashboard.py        # Streamlit Dashboard
│── api.py              # FastAPI API
│── requirements.txt
├── Dockerfile
├── startup.sh
└── README.md
```

---

## ⚙️ 1. Ejecución del Pipeline con Kedro

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/tu-repo.git
   cd tu-repo
   ```

2. Instala dependencias:
   ```bash
   pip install -r src/requirements.txt
   ```

3. Corre el pipeline de Kedro:
   ```bash
   kedro run
   ```

Esto generará:
- Un modelo LSTM entrenado (`modelo`)
- Datos procesados como `lstm_df_feat` y `dashboard_reservations`

---

## ☁️ 2. Subida a Azure Blob Storage

1. Crea un Storage Account desde el portal o CLI:
   ```bash
   az storage account create --name mystorage --resource-group mygroup --location eastus --sku Standard_LRS
   ```

2. Crea un contenedor:
   ```bash
   az storage container create --name datos --account-name mystorage
   ```

3. Sube los archivos:
   - `modelo/` → Carpeta con el modelo entrenado
   - `datos/lstm_df_feat.parquet` y `datos/dashboard_reservations.parquet`

Esto permite separar el cómputo del almacenamiento y mantener los datos **seguros, redundantes y cifrados**, conforme a las políticas de Azure.

---

## 🐳 3. Construcción y Push de la Imagen Docker

1. Construye la imagen Docker:
   ```bash
   docker build -t streamlitapp .
   ```

2. Loguéate en Azure Container Registry:
   ```bash
   az acr login --name tuacr
   ```

3. Etiqueta y sube la imagen:
   ```bash
   docker tag streamlitapp tuacr.azurecr.io/streamlitapp:v1
   docker push tuacr.azurecr.io/streamlitapp:v1
   ```

---

## 🚀 4. Despliegue en Azure Container Instance

1. Crea una instancia de contenedor:
   ```bash
   az container create      --resource-group mygroup      --name streamlit-container      --image tuacr.azurecr.io/streamlitapp:v1      --cpu 1      --memory 1.5      --registry-login-server tuacr.azurecr.io      --registry-username <usuario>      --registry-password <contraseña>      --ports 8000 8501      --dns-name-label streamlitappdemo      --environment-variables AZURE_STORAGE_CONNECTION_STRING="<conn_str>"
   ```

2. Accede al dashboard:
   - API: `http://streamlitappdemo.eastus.azurecontainer.io:8000`
   - Dashboard: `http://streamlitappdemo.eastus.azurecontainer.io:8501`

---

## 📌 Notas de Seguridad

- El modelo y los datos están almacenados en Azure Blob Storage, lo que asegura:
  - **Cifrado en reposo y en tránsito**
  - **Control de acceso**
  - **Redundancia y alta disponibilidad**
- Las variables sensibles como la cadena de conexión están ocultas en variables de entorno.

