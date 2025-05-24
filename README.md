# 📈 Real-Time Bitcoin Price Prediction Pipeline

This project implements a real-time Bitcoin price‐forecasting system using **BigDL** and **Apache Spark**, integrated with the **CoinGecko API** for live market data fetching.

---

## 🚀 Key Features

| Capability | Description |
|------------|-------------|
| 🔄 **ETL Pipeline** | Fetch, clean, transform, and store data with Spark |
| 🧠 **Distributed RNN Training** | LSTM model built & trained with BigDL on Spark |
| 🔮 **Forecasting** | Autoregressively predict future prices |
| 📊 **Visualization** | Matplotlib plots of *actual* vs *predicted* |
| 🐳 **Dockerized** | One-command reproducible environment |

---

## 🧰 Tech Stack

| Tool / Service | Role |
|----------------|------|
| **BigDL 2.4** | Distributed deep-learning library |
| **Apache Spark 3** | Large-scale data processing |
| **Matplotlib** | Result visualisation |
| **CoinGecko API** | Real-time Bitcoin price feed |
| **Docker** | Containerised runtime |

---

## 🗂️ Project Layout

| Path | Purpose |
|------|---------|
| `Bitcoin_pipeline.py` | End-to-end pipeline (ETL → training → prediction → plot) |
| `bitcoin_api.py` | Helpers to fetch & transform CoinGecko data |
| `Dockerfile` | Image definition (Spark + BigDL environment) |
| `docker_build.sh` | Build script (`docker build …`) |
| `docker_bash.sh` | Convenience shell (interactive Bash inside the image) |
| `docker_jupyter.sh` | Launch Jupyter Lab for ad-hoc exploration |
| `requirements.txt` | Pip dependencies for non-Docker runs |

---

## ⚙️ Running the Pipeline

> **Quick start (3 commands)**

```bash
# 1 – Build the image (tags it as `bigdl-bitcoin:latest`)
./docker_build.sh

# 2 – Start an interactive container, mounting the project
winpty docker run --rm -it \
       -v "$(pwd)":/app \
       -p 8888:8888 \
       bigdl-bitcoin:latest \
       bash

# 3 – Inside the container, execute the pipeline
python Bitcoin_pipeline.py
