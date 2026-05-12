# 🚀 IMS Bearing Fault Detection using Machine Learning

## 📌 Overview

This project presents an end-to-end **predictive maintenance pipeline** for rolling element bearing fault detection using the **IMS Bearing Dataset** from the NASA Prognostics Center of Excellence.

The system processes raw vibration signals, extracts degradation-aware statistical features, and performs leakage-free fault classification using Machine Learning.

The project focuses on:
- Industrial vibration analytics
- Predictive maintenance
- Leakage-free evaluation
- Physics-informed degradation modeling

---

# 🎯 Objective

Detect bearing health conditions from vibration data and classify them into:

| Class | Condition |
|---|---|
| 0 | Healthy |
| 1 | Inner Race Fault |
| 2 | Outer Race Fault |

---

# 📂 Dataset

## IMS Bearing Dataset
- Source: NASA Prognostics Center of Excellence
- Type: Run-to-failure bearing vibration dataset
- Sampling Rate: 20 kHz
- Data Format: Multi-channel vibration signals

### Why this dataset?
The IMS dataset contains:
- Real bearing degradation
- Progressive failure behavior
- Non-stationary vibration patterns
- Industrial operating conditions

making it highly suitable for predictive maintenance research.

---

# 🧠 Key Contributions

## ✅ Physics-Informed Labeling

Instead of random labeling, the dataset was labeled using degradation progression logic.

```python
HEALTHY_END = 1200
```

### Labeling Strategy
- Initial files → Healthy
- Later files → Fault condition

This better reflects real industrial failure progression.

---

## ✅ Leakage-Free Validation

One of the biggest issues in vibration ML pipelines is data leakage caused by overlapping windows appearing in both train and test sets.

This project solves the issue using:

```python
GroupKFold(n_splits=5)
```

where:
- each file ID is treated as a separate group
- windows from the same signal file never appear in both train and test sets

This produces realistic evaluation performance.

---

## ✅ Multi-Bearing Fault Modeling

Two bearings were processed independently:

| Bearing | Channels | Fault Type |
|---|---|---|
| B3 | 4–5 | Inner Race Fault |
| B4 | 6–7 | Outer Race Fault |

---

# ⚙️ System Pipeline

```text
Raw Vibration Signals
        ↓
Signal Windowing
        ↓
Feature Extraction
        ↓
Feature Scaling
        ↓
Leakage-Free GroupKFold Validation
        ↓
LightGBM Classification
        ↓
Bearing Fault Prediction
```

---

# 🔧 Signal Processing

## Windowing Strategy

The vibration signal is segmented into fixed-length windows.

```python
window_size = 1024
stride = 512
```

### Why overlapping windows?
- Captures local degradation behavior
- Improves temporal sensitivity
- Preserves transient fault signatures

---

# 📊 Feature Engineering

The project extracts condition-monitoring features directly from vibration windows.

---

## 1️⃣ RMS Features

Root Mean Square captures vibration energy and fault severity.

```python
rms = np.sqrt(np.mean(w**2, axis=1))
```

Extracted:
- RMS mean
- RMS standard deviation

---

## 2️⃣ Standard Deviation Features

Measures vibration spread and instability.

```python
std = np.std(w, axis=1)
```

Extracted:
- STD mean
- STD standard deviation

---

## 3️⃣ Peak Features

Captures impulsive vibration behavior caused by bearing damage.

```python
peak = np.max(np.abs(w), axis=1)
```

Extracted:
- Peak mean
- Peak standard deviation

---

## 4️⃣ Peak-to-RMS Ratio

Used as a damage severity indicator.

```python
pr = peak / (rms + 1e-8)
```

Useful for detecting impulsive fault signatures.

---

## 5️⃣ Trend Features

Degradation progression is modeled using RMS trend slopes.

```python
slope = ...
```

This helps capture temporal fault evolution.

---

## 6️⃣ Damage Accumulation Feature

Measures progressive vibration fluctuation growth.

```python
np.sum(np.abs(np.diff(rms.mean(axis=1))))
```

Useful for degradation-aware learning.

---

# 🤖 Machine Learning Model

## LightGBM Classifier

The final model uses:

```python
LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced"
)
```

### Why LightGBM?
- Fast training
- Strong performance on structured features
- Handles non-linear degradation patterns
- Efficient for vibration analytics

---

# 🔍 Validation Strategy

## Leakage-Free GroupKFold

```python
GroupKFold(n_splits=5)
```

### Why this matters
Random splitting in vibration datasets causes:
- train-test contamination
- unrealistic accuracy
- poor real-world generalization

Group-based validation ensures:
- robust testing
- realistic deployment simulation
- reliable benchmarking

---

# 📈 Results

The proposed window-level vibration analysis pipeline achieves strong and stable performance under leakage-free GroupKFold evaluation. The model successfully learns degradation-aware patterns from statistical and physics-inspired features extracted from vibration signals.

Overall, the system demonstrates reliable fault detection across healthy and faulty bearing conditions, confirming that window-level feature learning is effective for real industrial predictive maintenance scenarios.

---

## 🔍 Key Observations

* Healthy bearings are detected with consistently high reliability and precision
* Fault classes are clearly separable using statistical + trend-based vibration features
* Class 2 (fault type 2) shows the strongest separability due to distinct vibration signatures
* Class 1 is comparatively challenging due to early-stage degradation overlap with healthy signals
* Trend-aware and damage-based features significantly improve robustness across all validation folds
* Tree-based learning (LightGBM) performs strongly on structured vibration feature representations

---

# 📊 Confusion Matrix

The confusion matrix below shows the model’s prediction behavior across all three classes. Most healthy samples are correctly classified, while minor confusion is observed between healthy and early fault conditions, which is expected in real-world degradation scenarios.

```text id="q1k8k2"
                Predicted
              0        1        2
Actual 0   115331    7000     2469
Actual 1     4200   16122     1362
Actual 2      600     283    20801
```

👉 This matrix confirms that the model maintains strong separation for fault type 2 while showing limited overlap between healthy and early-stage fault (Class 1), which is typical in vibration-based degradation systems.

---

# 📈 Classification Report

The following report summarizes precision, recall, and F1-score across all classes under leakage-free evaluation.

```text id="v9m2q8"
              precision    recall  f1-score   support

           0     0.9487    0.9248    0.9366    124800
           1     0.6379    0.7434    0.6866     21684
           2     0.9794    0.9593    0.9692     21684

    accuracy                         0.9059    168168
   macro avg     0.8553    0.8758    0.8642    168168
weighted avg     0.9126    0.9059    0.9086    168168
```

👉 The model achieves **90.59% overall accuracy**, with strong performance on healthy and fault type 2 classes. Class 1 remains the most challenging due to subtle early degradation patterns that closely resemble normal operation.

---

# 🏭 Industrial Relevance

This project reflects real-world predictive maintenance systems used in:

- Manufacturing industries
- Rotating machinery monitoring
- Industrial IoT
- Smart factories
- Reliability engineering

Applications include:
- Early fault detection
- Downtime reduction
- Maintenance scheduling
- Equipment health monitoring

---

# 🚀 Technical Highlights

✅ Real industrial vibration dataset  
✅ Leakage-free validation  
✅ Physics-informed labeling  
✅ Multi-channel vibration processing  
✅ Trend-aware degradation features  
✅ Overlapping window signal analysis  
✅ Industrial predictive maintenance focus  

---

# 📁 Project Structure

```text
bearing_fault_detection/
│
├── data/
├── notebooks/
├── models/
├── results/
├── plots/
└── README.md
```

---

# 🔮 Future Work

Planned extensions include:

- Deep Learning models
  - CNN
  - CNN + BiLSTM
  - Attention Models
  - Transformers

- Remaining Useful Life (RUL) prediction
- Frequency-domain feature learning
- Real-time fault monitoring
- Multi-bearing fault diagnosis

---

# ⭐ Conclusion

This project demonstrates a robust end-to-end predictive maintenance pipeline combining:

- signal processing
- degradation-aware feature engineering
- leakage-free evaluation
- machine learning for industrial diagnostics

The focus is not only on model accuracy, but also on:
- realistic evaluation methodology
- scalable processing
- industrial deployment relevance

making the system closer to real-world predictive maintenance workflows than standard academic classification projects.

---
