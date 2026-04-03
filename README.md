# bearing_fault_detection

Got it — now I’ll compress your full work into a **clean, professional, high-impact README** that keeps **all important technical depth** but removes unnecessary overload.

This is exactly what recruiters + GitHub viewers want 👇

---

# 🚀 Bearing Fault Detection using Machine Learning

## 📌 Overview

This project builds an **end-to-end machine learning pipeline** for detecting faults in rolling element bearings using vibration signals.

The system classifies bearing condition into:

* **Healthy**
* **Inner Race Fault**
* **Outer Race Fault**
* **Ball Fault**

It combines **signal processing + physics-informed labeling + ML models** to achieve high accuracy on real-world data.

---

## 📂 Dataset

* **Dataset:** IMS Bearing Dataset
* **Source:** NASA Prognostics Center of Excellence
* **Sampling Rate:** 20 kHz
* **Type:** Run-to-failure vibration data
* **Format:** Multiple multi-channel CSV files

### Why this dataset?

* Real degradation (not synthetic)
* Faults evolve over time
* Suitable for **predictive maintenance systems**

---

## 🧠 Key Contributions

### ✅ Physics-Informed Labeling

* First ~80% of each file → **Healthy**
* Remaining → **Fault type (based on test setup)**

→ Mimics real fault progression instead of random labeling

---

### ✅ Leakage-Free Evaluation

* Used **GroupKFold (group = file ID)**
* Prevents windows from same signal appearing in both train & test

---

### ✅ Large-Scale Data Handling

* ~369K signal windows
* Multi-file → single structured dataset

---

## ⚙️ Pipeline

### 1. Signal Processing

* Window size: **1024**
* Stride: **512 (overlapping)**

---

### 2. Feature Engineering

#### Time Domain

* Mean, RMS, Std, Skewness, Kurtosis
* Crest factor, Impulse factor, Shape factor

#### Frequency Domain

* FFT-based features
* Spectral centroid, bandwidth, entropy
* Dominant frequency

#### Band Energy Features

* 0–1kHz, 1–5kHz, 5–10kHz, 10kHz+

#### Envelope Features

* Hilbert transform
* Envelope RMS

---

### 3. Feature Processing

* StandardScaler
* LightGBM-based feature selection

---

### 4. Models Used

* **LightGBM** (Best)
* **Random Forest**
* **XGBoost**
* **SVM**
* **KNN (with PCA)**

---

## 📊 Results

### 🔥 Cross-Validation Accuracy (5-Fold GroupKFold)

| Model         | Mean Accuracy |
| ------------- | ------------- |
| LightGBM      | **~96.7%**    |
| Random Forest | ~96.4%        |
| XGBoost       | ~96.2%        |
| KNN           | ~89.7%        |
| SVM           | ~87.5%        |

---

### 📌 Classification Highlights (LightGBM)

* **Inner Race:** Perfect detection (F1 = 1.00)
* **Outer Race:** ~0.98 F1
* **Ball Fault:** ~0.96 F1
* **Healthy:** ~0.97 F1

---

## 📈 Key Insights

* Tree-based models significantly outperform SVM/KNN
* Feature engineering (time + frequency + envelope) is critical
* Physics-based labeling improves real-world relevance
* GroupKFold is essential to avoid data leakage

---

## 🏁 Conclusion

This project demonstrates a **production-level ML pipeline** for predictive maintenance:

* Raw vibration → structured features
* Smart labeling strategy
* Robust evaluation
* High accuracy (~97%)

---

## 🔮 Future Work

* Deep Learning (CNN / LSTM / Transformers)
* Real-time deployment
* Multi-component fault diagnosis

---

## ⭐ Why This Project Stands Out

* Real industrial dataset
* Physics + ML integration
* Leakage-aware validation
* Multi-model benchmarking

---

If you want next level:
I can make this into a **🔥 standout GitHub (with diagrams + visuals)** or a **LinkedIn post series that actually gets inbound DMs** — that’s where the real impact comes.
