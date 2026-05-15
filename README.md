
# 🚀 IMS Bearing Fault Detection using Machine Learning

## 📌 Overview

This project presents an end-to-end **predictive maintenance system** for rolling element bearing fault detection using the IMS Bearing Dataset from NASA Prognostics Center of Excellence.

The system converts raw vibration signals into meaningful multi-domain representations and applies machine learning for fault classification under a leakage-free evaluation setup.

The focus is on:

* Vibration-based condition monitoring
* Multi-domain feature engineering
* Data-driven feature selection
* Leakage-free model evaluation
* Industrial predictive maintenance applications

---

# 🎯 Objective

Detect bearing health conditions from vibration data and classify them into:

| Class | Condition        |
| ----- | ---------------- |
| 0     | Healthy          |
| 1     | Inner Race Fault |
| 2     | Ball Race Fault |

---

# 📂 Dataset

## IMS Bearing Dataset

* Source: NASA Prognostics Center of Excellence
* Type: Run-to-failure vibration dataset
* Sampling Rate: 20 kHz
* Multi-channel accelerometer signals

### Why this dataset?

It captures real-world degradation patterns including:

* gradual fault progression
* non-stationary vibration behavior
* realistic industrial noise conditions

---

# ⚙️ System Pipeline

The system follows a structured multi-stage processing flow:

```text
Raw Vibration Signal
        ↓
Windowing (Fixed-Length Segments)
        ↓
Multi-Domain Feature Extraction
        ↓
LightGBM Feature Importance Selection
        ↓
Top-K Feature Subset
        ↓
Final Classification (XGBoost / Random Forest)
```

---

# 🔧 Signal Processing

## Windowing Strategy

The continuous vibration signal is segmented into overlapping windows:

```python
window_size = 1024
stride = 512
```

### Purpose of windowing:

* Capture local degradation behavior
* Preserve transient fault signatures
* Increase temporal resolution of analysis

---

# 📊 Feature Engineering

The system extracts features from multiple signal perspectives.

---

## 📈 Time-Domain Statistical Features

Basic vibration descriptors:

* RMS → vibration energy level
* Standard deviation → signal variability
* Peak amplitude → impulsive intensity
* Peak-to-RMS ratio → fault impulsiveness

---

## 📉 Trend & Degradation Features

* Slope of signal evolution → captures gradual degradation behavior
* Damage accumulation → measures progressive vibration changes over time
* Roughness → captures micro-level instability in signal evolution

---

## ⚡ Frequency-Domain Features

* High-frequency energy → captures energy shift during fault formation
* Spectral entropy → measures disorder in frequency distribution

---

## 📡 Envelope-Based Features

* Envelope energy (Hilbert transform) → highlights hidden modulation patterns caused by repetitive impacts

---

## 💥 Impulsive & Distribution Features

* Crest factor → peak-to-average vibration intensity
* Kurtosis → heavy-tailed impulsive behavior
* Skewness → asymmetry in vibration distribution

---

# 🌲 Feature Selection (LightGBM Stage)

A tree-based model is used to rank feature importance:

```python
LGBMClassifier → Feature Importance → Top-K Selection
```

### Purpose:

* Remove redundant signal representations
* Retain most discriminative features
* Improve generalization across folds
* Reduce noise in high-dimensional feature space

---

# 🤖 Machine Learning Models

The selected feature subset is passed to:

* XGBoost Classifier (primary model)
* Random Forest Classifier (benchmark model)

Both models operate on identical selected feature sets for fair comparison.

---

# 🔍 Validation Strategy

## Leakage-Free GroupKFold

```python
GroupKFold(n_splits=5)
```

### Why this matters:

* Ensures no overlap of windows from same file across train/test
* Prevents data leakage in time-series vibration data
* Produces realistic industrial evaluation

---

# 📈 Key Observations

* Healthy condition shows stable vibration patterns with high separability
* Fault classes exhibit distinct impulsive and spectral characteristics
* Early-stage faults are harder to distinguish due to overlapping behavior with healthy signals
* Feature selection significantly improves stability across folds
* Tree-based models perform strongly on structured vibration representations

---

# 🏭 Industrial Relevance

This system is applicable to:

* Predictive maintenance systems
* Industrial IoT monitoring
* Rotating machinery diagnostics
* Smart factory condition monitoring
* Reliability engineering workflows

Use cases include:

* Early fault detection
* Maintenance scheduling optimization
* Equipment health monitoring

---

# 🚀 Technical Highlights

* Multi-domain vibration feature engineering
* FFT + entropy + envelope + impulsive signal analysis
* LightGBM-based feature selection
* XGBoost and Random Forest benchmarking
* Leakage-free GroupKFold evaluation
* Real industrial dataset (IMS NASA)

---

# 🔮 Future Work

* Deep learning models (CNN, LSTM, Transformers)
* Remaining Useful Life (RUL) prediction
* Real-time streaming fault detection
* Multi-sensor fusion
* Adaptive online learning systems

---

# ⭐ Conclusion

This project demonstrates a complete predictive maintenance pipeline built around real vibration data, focusing on:

* meaningful signal representation
* data-driven feature selection
* robust evaluation strategy
* industrial applicability

Rather than relying solely on model complexity, the system emphasizes structured transformation of raw vibration signals into meaningful representations for fault diagnosis.

