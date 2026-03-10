# A Hybrid Deep Learning Approach to Stress Detection
### Integrating CNN-LSTM with Reinforcement Learning and Active Learning

![Python](https://img.shields.io/badge/Python-DeepLearning-blue)
![Framework](https://img.shields.io/badge/Framework-TensorFlow-green)
![Paper](https://img.shields.io/badge/Paper-Springer-orange)

Official implementation of the research paper:

**“A Hybrid Deep Learning Approach to Stress Detection: Integrating CNN-LSTM with Reinforcement Learning and Active Learning.”**

This repository provides the implementation of a hybrid deep learning framework designed for **stress detection using physiological signals**.

---

# 📄 Publication

This repository accompanies the following publication:

**A Hybrid Deep Learning Approach to Stress Detection: Integrating CNN-LSTM with Reinforcement Learning and Active Learning**

Published in **Springer – Lecture Notes in Networks and Systems (LNNS)**.

Paper link:

https://www.researchgate.net/publication/399379127_A_Hybrid_Deep_Learning_Approach_to_Stress_Detection_Integrating_CNN-LSTM_with_Reinforcement_Learning_and_Active_Learning

---

# 🧠 Method Overview

The proposed framework integrates three major components:

### CNN Feature Extraction

Convolutional Neural Networks extract spatial features from physiological signals.

### LSTM Temporal Modeling

Long Short-Term Memory networks capture temporal dependencies and dynamic stress patterns.

### Reinforcement Learning (DDQN)

A **Double Deep Q Network (DDQN)** is used to adaptively improve model decision strategies.

### Active Learning

Active learning selects the most informative samples, reducing labeling cost and improving learning efficiency.

---

# 🏗 Architecture

Pipeline overview:

Physiological Signals  
↓  
CNN Feature Extraction  
↓  
LSTM Temporal Modeling  
↓  
Reinforcement Learning (DDQN)  
↓  
Active Learning Sample Selection  
↓  
Stress Classification  

---

# 📊 Results

| Model | Accuracy |
|------|------|
| CNN-LSTM Baseline | 94.10% |
| CNN-LSTM + Reinforcement Learning + Active Learning | **95.10%** |

The hybrid learning framework improves performance by combining **deep temporal modeling with adaptive learning strategies**.

---

# 📁 Repository Structure
```
A-Hybrid-Deep-Learning-Approach-to-Stress-Detection
│
├── src
│    └── Model
│          ├── wesad-chest-resp-hybrid.py
│          ├── wesad-chest-resp-imb-hyb-al.py
│          ├── wesad-chest-resp-imb-hyb-dqn-td-reward-2labels-al-l1-norm.py
│          ├── wesad-chest-resp-imb-hyb-dqn.py
│          └── wesad-chest-resp.py
│
│
├── requirements.txt
└── README.md
```
# ⚙️ Installation

Clone repository
```
git clone https://github.com/DSTai/A-Hybrid-Deep-Learning-Approach-to-Stress-Detection.git
```
Install dependencies
```
pip install -r requirements.txt
```
# ▶️ Training

Run training script
```
python src/train.py
```
# 💡 Applications

This research can contribute to:

- Wearable stress monitoring
- Mental health monitoring systems
- Human affective computing
- AI-driven healthcare
- Physiological signal analysis

---

# 🔬 Future Work

Possible improvements:

- Transformer-based time series models
- Multimodal physiological data fusion
- Edge AI deployment on wearable devices
- Real-time stress monitoring systems

---

# 👨‍💻 Author

**HDT Tran**

AI / Machine Learning Researcher  
Deep Learning for Physiological Signal Processing

GitHub  
https://github.com/DSTai

