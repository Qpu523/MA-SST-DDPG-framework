# Modeling Interactive Crash Avoidance Behaviors: A Multi-Agent State-Space Transformer-Enhanced Reinforcement Learning Framework

This repository hosts the implementation and evaluation of the **Multi-Agent State-Space Transformer Deep Deterministic Policy Gradient (MA-SST-DDPG)** framework, designed to model **interactive crash avoidance behaviors** between vehicles and pedestrians in near-miss scenarios at urban intersections.

---

## 🧠 Framework Overview

The **MA-SST-DDPG framework** integrates:

- **Multi-Agent DDPG:** Captures cooperative–competitive dynamics between vehicles and pedestrians using centralized training with decentralized execution.  
- **State-Space Model (Mamba):** Efficiently models long-term temporal dependencies in sequential decision-making.  
- **Transformer Module:** Dynamically prioritizes safety-critical features within interaction histories.  

<p align="center">
  <img src="https://github.com/Qpu523/MA-SST-DDPG-framework/blob/56f7e91bb91b5e90c4dfe53414551139603b4d51/Config/F1.png" alt="Framework Overview" width="600">
</p>

This hybrid design enables accurate reconstruction of evasive maneuvers and realistic simulation of human-like crash avoidance behaviors.

---

## 📂 Dataset

Training and evaluation were conducted on the **High-Density Intersection (HDI) UAV dataset** collected in Hohhot, China:

- **Duration:** 4 hours of 4K drone footage  
- **Resolution:** 3840×2160, 0.1 s temporal resolution  
- **Detection & Tracking:** YOLOv8 + Bytetrack/BoT-SORT with Kalman Filter and GMC corrections  
- **Coverage:** >330 identified near-miss interactions (CurvTTC < 5s)  

Additional external datasets for cross-validation:

- **Argoverse 2** (urban trajectories from six U.S. cities)  
- **TGSIM** (Third-Generation Simulation dataset: highways + cities in Chicago and Washington, D.C.)  

Dataset access: [High-Density Intersection Dataset](https://github.com/Qpu523/HDI-Dataset)

---

## ⚙️ Methodology

- **Environment:** Vehicle and pedestrian modeled as agents in a two-agent Markov game.  
- **Action Space:** Continuous (longitudinal and lateral accelerations).  
- **Replay Buffer & Reweighting:** Importance sampling with TD error and feedback weights to emphasize critical near-miss cases.  
- **Training Parameters:**  
  - Buffer size: 10,000  
  - Batch size: 256  
  - Discount factor: 0.9  
  - Target update rate: 0.01  
  - 3,000 training episodes  

---

## 📊 Results

### Reconstruction Accuracy

- Vehicle longitudinal speed RMSE: **0.058 m/s**  
- Pedestrian longitudinal speed RMSE: **0.056 m/s**  
- ADE: **0.078 m**, FDE: **0.156 m**  

<p align="center">
  <img src="https://github.com/Qpu523/MA-SST-DDPG-framework/blob/56f7e91bb91b5e90c4dfe53414551139603b4d51/Config/F2.png" alt="Trajectory Reconstruction" width="600">
</p>

### Model Comparison
<p align="center">
<img src="https://github.com/Qpu523/MA-SST-DDPG-framework/blob/56f7e91bb91b5e90c4dfe53414551139603b4d51/Config/table1.png" alt="Model Compare" width="600">
</p>

MA-SST-DDPG consistently outperforms both **trajectory forecasting baselines** and **reinforcement learning variants**, particularly in safety-critical conditions.

---

## 🔬 Safety Analysis with Generated Data

Using the trained MA-SST-DDPG, we simulated **5,102 high-risk interactions** under varied initial conditions:

- **Conflict Rate:** Increased sharply when both vehicle and pedestrian initial speeds exceeded 3.5 m/s.  
- **Avoidance Strategies:**  

  - When **vehicle speed > 3.2 m/s** and **pedestrian speed < 1.9 m/s**, **Pedestrian-Yield** dominated (89.8%), indicating that fast vehicles with slow pedestrians led pedestrians to yield.  
  - When **vehicle speed < 3.2 m/s** and **pedestrian speed ≥ 1.9 m/s**, **Vehicle-Yield** dominated (90.7%), showing that faster pedestrians with slower vehicles typically triggered vehicle yielding.  
  - In the **low–low speed quadrant**, Vehicle-Yield was more frequent (62.5%).  
  - In the **high–high speed quadrant**, Pedestrian-Yield was slightly more frequent (57.1%).  

| Conflict Rate Surface | Yield Strategy Quadrant |
|-----------------------|--------------------------|
| <img src="https://github.com/Qpu523/MA-SST-DDPG-framework/blob/56f7e91bb91b5e90c4dfe53414551139603b4d51/Config/F3.png" width="400"/> | <img src="https://github.com/Qpu523/MA-SST-DDPG-framework/blob/56f7e91bb91b5e90c4dfe53414551139603b4d51/Config/F4.png" width="400"/> |


---


