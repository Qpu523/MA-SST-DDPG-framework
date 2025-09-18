# Modeling Interactive Crash Avoidance Behaviors: A Multi-Agent State-Space Transformer-Enhanced Reinforcement Learning Framework

This repository hosts the implementation and evaluation of the **Multi-Agent State-Space Transformer Deep Deterministic Policy Gradient (MA-SST-DDPG)** framework, designed to model **interactive crash avoidance behaviors** between vehicles and pedestrians in near-miss scenarios at urban intersections.

---

## 🧠 Framework Overview

The **MA-SST-DDPG framework** integrates:

- **Multi-Agent DDPG:** Captures cooperative–competitive dynamics between vehicles and pedestrians using centralized training with decentralized execution.  
- **State-Space Model (Mamba):** Efficiently models long-term temporal dependencies in sequential decision-making.  
- **Transformer Module:** Dynamically prioritizes safety-critical features within interaction histories.  

<p align="center">
  <img src="https://github.com/Qpu523/MA-SST-DDPG-framework/blob/792ab2973d99daded87bddee3babacd4228bde36/Config/Picture2.png" alt="Framework Overview" width="600">
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
- **Reward Functions:**
  - **Speed Reward** → aligns agent velocities with human-like behavior.  
  - **Distance Reward** → maintains realistic separation during interactions.  
- **Replay Buffer & Reweighting:** Importance sampling with TD error and feedback weights to emphasize critical near-miss cases.  
- **Training Parameters:**  
  - Buffer size: 10,000  
  - Batch size: 256  
  - Actor LR: 5e-4, Critic LR: 1e-3  
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
  <img src="https://github.com/Qpu523/MA-SST-DDPG-framework/blob/cf4c0a9b9a7cfa45763bd13782795c8966fc4db8/Config/Picture3.png" alt="Trajectory Reconstruction" width="600">
</p>

### Model Comparison

| Category                     | Model                  | RMSE v_veh^y (m/s) | RMSE v_veh^x (m/s) | RMSE v_ped^y (m/s) | RMSE v_ped^x (m/s) | RMSE d_vp^y (m) | RMSE d_vp^x (m) | RMSE d_pv^y (m) | RMSE d_pv^x (m) | ADE (m) | FDE (m) | CT (ms/step) |
|------------------------------|------------------------|--------------------|--------------------|--------------------|--------------------|-----------------|-----------------|-----------------|-----------------|---------|---------|--------------|
| **Proposed model**           | **MA-SST-DDPG**        | **0.058** | **0.023** | **0.056** | **0.030** | **0.098** | **0.118** | **0.103** | **0.126** | **0.078** | **0.156** | 16 |
| **Ablation Experiments**     | MA-State-Space-DDPG    | 0.062 | 0.024 | 0.058 | 0.035 | 0.108 | 0.122 | 0.109 | 0.133 | 0.083 | 0.165 | 15 |
|                              | MA-Transformer-DDPG    | 0.061 | 0.026 | 0.057 | 0.035 | 0.145 | 0.136 | 0.146 | 0.135 | 0.098 | 0.196 | 18 |
|                              | MA-LSTM-DDPG           | 0.098 | 0.027 | 0.088 | 0.042 | 0.181 | 0.146 | 0.210 | 0.166 | 0.123 | 0.247 | 22 |
|                              | SST-DDPG               | 0.132 | 0.039 | 0.135 | 0.050 | 0.312 | 0.152 | 0.210 | 0.193 | 0.152 | 0.304 | 9 |
|                              | State-Space-DDPG       | 0.184 | 0.078 | 0.230 | 0.056 | 0.382 | 0.342 | 0.412 | 0.307 | 0.253 | 0.505 | 8 |
|                              | Transformer-DDPG       | 0.264 | 0.079 | 0.233 | 0.086 | 0.383 | 0.387 | 0.434 | 0.346 | 0.271 | 0.542 | 10 |
| **Reinforcement Learning**   | MA-LSTM-DDPG           | 0.098 | 0.027 | 0.088 | 0.042 | 0.181 | 0.146 | 0.210 | 0.166 | 0.123 | 0.247 | 22 |
|                              | MA-DDPG                | 0.147 | 0.045 | 0.134 | 0.048 | 0.322 | 0.184 | 0.227 | 0.203 | 0.164 | 0.328 | 14 |
|                              | DDPG                   | 0.594 | 0.313 | 0.802 | 0.181 | 1.001 | 1.050 | 0.656 | 0.671 | 0.679 | 1.357 | 7 |
| **Trajectory Prediction**    | DenseTNT               | 0.315 | 0.180 | 0.350 | 0.210 | 0.450 | 0.380 | 0.480 | 0.410 | 0.420 | 0.850 | 12 |
|                              | MultiPath++            | 0.340 | 0.195 | 0.380 | 0.230 | 0.490 | 0.410 | 0.520 | 0.440 | 0.450 | 0.920 | 14 |
|                              | Social-LSTM            | 0.450 | 0.250 | 0.520 | 0.310 | 0.680 | 0.550 | 0.710 | 0.580 | 0.650 | 1.350 | 8 |
| **Supervised Learning**      | Transformer            | 1.156 | 0.438 | 1.661 | 0.261 | 1.751 | 1.169 | 1.549 | 1.624 | 1.199 | 2.398 | 6 |
|                              | LSTM                   | 2.066 | 1.877 | 2.059 | 2.011 | 2.024 | 1.847 | 2.128 | 2.012 | 1.783 | 2.566 | 7 |
|                              | NN                     | 2.252 | 2.256 | 2.145 | 2.382 | 2.195 | 2.145 | 2.055 | 2.008 | 2.677 | 3.955 | 4* |

*The best values among all models are highlighted in bold. ADE/FDE are averaged over vehicle and pedestrian evaluation sequences.*


MA-SST-DDPG consistently outperforms both **supervised trajectory forecasting baselines** and **reinforcement learning variants**, particularly in safety-critical conditions:contentReference[oaicite:0]{index=0}.

---

## 🔬 Safety Analysis with Generated Data

Using the trained MA-SST-DDPG, we simulated **5,102 high-risk interactions** under varied initial conditions:

- **Conflict Rate:** Increased sharply when both vehicle and pedestrian initial speeds exceeded 3.5 m/s.  
- **Avoidance Strategies:**
  - **Vehicle-Yield (67.3%)** → braking/deceleration dominant.  
  - **Pedestrian-Yield (32.7%)** → slowing down or rollback to allow vehicles to pass.  

<p align="center">
  <img src="docs/conflict_rate.png" alt="Conflict Rate Surface" width="500">
  <img src="docs/strategy.png" alt="Yield Strategy Quadrant" width="500">
</p>

---

## 📥 Code & Usage

1. Clone this repository:  
   ```bash
   git clone https://github.com/your-username/CrashAvoidance_MA-SST-DDPG.git
   cd CrashAvoidance_MA-SST-DDPG
