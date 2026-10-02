# H2-Grid India

> **A Probabilistic, Physics-Grounded Forecasting and Techno-Economic Siting Engine for Solar-Driven Green Hydrogen Production**

---

## 📌 Project Overview

**H2-Grid India** is an end-to-end forecasting and techno-economic siting platform designed to evaluate green hydrogen production feasibility across India's renewable energy corridors. 

The project places an empirical spotlight on **South Indian Renewable Corridors**—evaluating high-irradiance semi-arid zones against severe Western Ghats monsoon cloud attenuation—while maintaining coordinate-agnostic Pan-India scalability:
* **Palakkad, Kerala** ($10.78^\circ\text{N}, 76.65^\circ\text{E}$): Wet tropical / Western Ghats monsoon stress-test.
* **Ramanathapuram, Tamil Nadu** ($9.36^\circ\text{N}, 78.83^\circ\text{E}$): Tropical coastal green hydrogen export corridor.
* **Pavagada, Karnataka** ($14.10^\circ\text{N}, 77.28^\circ\text{E}$): Semi-arid Deccan Plateau mega solar park.
* **Bhadla, Rajasthan** ($27.53^\circ\text{N}, 71.91^\circ\text{E}$): National benchmark; highest direct normal irradiance (DNI).

---

## ⚡ Core Architecture at a Glance

```text
NASA POWER API (5-Yr Hourly Data)
       │
       ▼
Solar Astronomy & Feature Engineering (Zenith, Cyclical Time, Lags, Monsoon Flags)
       │
       ▼
4-Tier Model Hierarchy (SARIMA ➔ BiLSTM ➔ LightGBM ➔ Quantile TFT)
       │
       ▼
Electrochemical Digital Twin (PV Derating + PEM Polarization & Faraday Efficiency)
       │
       ▼
Techno-Economic Engine (Monte Carlo LCOH vs National Green Hydrogen Mission Target)
       │
       ▼
Full-Stack Interactive Dashboard (FastAPI + SQLite + React + Vite + Leaflet)
```

---

## 🚀 Environment Setup & Quickstart

### 1. Prerequisites
* **Python:** Version 3.10, 3.11, or 3.12 installed.
* **Git:** Installed and configured.

### 2. Virtual Environment Setup
Open PowerShell in the project root (`C:\Dev\H2 grid`) and run:

```powershell
# 1. Create the isolated virtual environment
python -m venv .venv

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```
*(Once activated, you will see `(.venv)` in your terminal prompt).*

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Fetch Multi-Year Benchmark Datasets
To extract 5 years ($2019–2023$) of hourly satellite solar and meteorological data for all benchmark hubs:
```powershell
python -m src.data.download_hubs
```
*(Downloaded raw datasets will be stored in `data/raw/`).*

---

> **Note:** Comprehensive architecture schematics, full API documentation, and comparative ablation tables will be compiled upon completion of Phase 8.
