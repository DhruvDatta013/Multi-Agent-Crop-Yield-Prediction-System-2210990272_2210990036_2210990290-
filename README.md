# Multi-Agent-Crop-Yield-Prediction-System-2210990272_2210990036_2210990290
Multi-agent AI system for crop yield prediction and smart agricultural decision support using machine learning.

---

## 📋 Project Information

- **Project Title:** An Intelligent Multi-Agent Framework for Enhanced Crop Yield Prediction and Agricultural Decision Support
- **Type:** Research
- **Current Status:** ✅ Completed
- **Institution:** Chitkara University Institute of Engineering and Technology, Punjab, India
- **Department:** Computer Science and Engineering
- **Programme:** Bachelor of Engineering – Computer Science and Engineering

---

## 👥 Team Details

| Roll Number | Name | Email |
|-------------|------|-------|
| 2210990272 | Dhruv Datta | dhruvdatta11@gmail.com |
| 2210990036 | Abhinav Bhar | abhinavbhar2004@gmail.com |
| 2210990290 | Divanshu Kaushik | divanshukaushik123@gmail.com |

---

## 🧑‍🏫 Mentor

| Name | Designation |
|------|-------------|
| Dr. Gurpreet Singh | Mentor, Faculty – Dept. of CSE, Chitkara University |

---

## 📖 About the Project

Global food security is one of the most critical challenges of the 21st century. With the global population projected to exceed 9.7 billion by 2050, there is a growing need for accurate and scalable crop yield prediction systems that can guide farmers, policy-makers, and agri-businesses in making data-driven decisions.

This project proposes an **intelligent five-agent multi-agent system (MAS)** for end-to-end crop yield prediction and agricultural decision support. Unlike conventional single-model approaches, the system decomposes the entire agricultural pipeline into five specialized, cooperative agents — each handling a specific phase of the workflow and communicating through a shared Knowledge Base (blackboard architecture).

The framework was evaluated on the **FAO Crop Yield Prediction Dataset** (Kaggle), containing 28,242 raw records across 101 countries and 10 crop types spanning 1990–2013. After deduplication, 25,932 clean records were used for training and evaluation.

---

## 🤖 System Architecture – Five Agent Pipeline

| Agent | Role |
|-------|------|
| **Agent 1 – Data Collection Agent** | Loads raw CSV data, removes duplicates, validates records, stores cleaned data in the Knowledge Base |
| **Agent 2 – Data Preprocessing Agent** | Applies one-hot encoding, kNN imputation, Min-Max normalization, and stratified 70:30 train-test split |
| **Agent 3 – Learning Agent** | Trains four ML models (Linear Regression, Random Forest, Agent-RF, SVM), selects best model by minimum RMSE |
| **Agent 4 – Decision-Making Agent** | Converts yield predictions into HIGH / MEDIUM / LOW advisory tiers with irrigation, fertilization, and harvesting recommendations |
| **Agent 5 – Feedback Agent** | Monitors MAE and RMSE thresholds, triggers adaptive retraining with escalating hyperparameters (up to 5 cycles) |

All agents read from and write to a central **Knowledge Base**, enabling modular communication and adaptive feedback loops.

---

## 📊 Key Results

| Model | Accuracy (%) | MAE (hg/ha) | RMSE (hg/ha) | R² |
|-------|:---:|:---:|:---:|:---:|
| Linear Regression (Baseline) | 21.3% | 30,407 | 43,665 | 0.745 |
| Random Forest (No Agent) | 84.8% | 4,099 | 10,335 | 0.986 |
| **Agent-Based Random Forest (Proposed)** | **84.6%** | **4,078** | **10,252** | **0.986** |
| Agent-Based SVM (LinearSVR) | 9.5% | 62,402 | 104,357 | -0.458 |

- The **Agent-Based Random Forest** achieved the best overall performance with **R² = 0.986**, outperforming Linear Regression by **64.9 percentage points** in accuracy.
- The Feedback Agent triggered **5 retraining cycles**, demonstrating stable convergence of the adaptive loop.
- The Decision-Making Agent classified **58.3% HIGH**, **26.4% MEDIUM**, and **15.2% LOW** yield predictions, achieving a **84.8% high-confidence advisory rate**.

---

## 🔍 Top Features (Agent-Based Random Forest)

1. Item_Potatoes
2. Item_Cassava
3. pesticides_tonnes
4. Item_Soybeans
5. Item_Wheat
6. avg_temp
7. average_rain_fall_mm_per_year

---

## 📈 Comparison with Related Work

| Reference | Method | Dataset | Best R² | Accuracy |
|-----------|--------|---------|---------|----------|
| Gandhi et al. | Random Forest | India (Rice) | 0.921 | ~85% |
| Pantazi et al. | SVM / ANN | EU Wheat | 0.890 | ~82% |
| Van Klompenburg | RF (review avg.) | Multi-crop | 0.903 | – |
| Nevavuori et al. | CNN-LSTM | Finland Wheat | 0.941 | ~88% |
| **Proposed Framework** | **Agent-Based RF** | **FAO 101 countries** | **0.986** | **84.6%** |

---

## 🛠️ Technologies Used

- **Language:** Python 3.10
- **ML Libraries:** scikit-learn, NumPy, Pandas
- **Visualisation:** Matplotlib, Seaborn
- **Dataset:** FAO Crop Yield Prediction Dataset (Kaggle)
- **Development Tools:** Jupyter Notebook, VS Code, Git & GitHub

---

## 🗂️ Report Structure

```
├── Declaration
├── Acknowledgement
├── List of Figures and Tables
├── Abstract
├── Chapter 1 – Introduction
├── Chapter 2 – Methodology
├── Chapter 3 – Tools and Technologies
├── Chapter 4 – Implementation
├── Chapter 5 – Major Findings / Outcomes / Results
├── Chapter 6 – Conclusion and Future Scope
└── References
```

---

## 🚀 Future Scope

- Integration of satellite imagery (NDVI, EVI) for spatial yield prediction
- LSTM-based Temporal Forecasting Agent for multi-year planning
- Anomaly Detection Agent for outlier climate years
- Real-time IoT sensor data integration for on-farm deployment
- Web-based farmer-facing advisory dashboard
- Federated learning for privacy-preserving cross-farm model training
- Blockchain integration for secure agricultural data management

---

## ✅ Status

**COMPLETED** — Submitted as COOP-II (Project at Industry, Module-2) 
