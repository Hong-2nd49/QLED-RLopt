# QLED-RLopt
QLED-RLopt is a research-oriented toolkit that uses "reinforcement learning (RL)" and "physics-based simulation" to optimize the architecture of quantum dot light-emitting diodes (QLEDs).

**Reinforcement Learning Optimization Framework for Microstructured Quantum Dot LED Devices**  
用于微结构量子点发光二极管（QLED）器件结构优化的强化学习框架

---

## 1. Overview | 项目简介

QLED-RLopt is a research-oriented toolkit that uses **reinforcement learning (RL)** and **physics-based or surrogate simulations** to optimize QLED device architectures.

Core goals:

- Explore **layered & micro/nano-structured QLED stacks**  
  (e.g. lateral ZnO–QD patterns, multi-EML, interlayers)
- Consider **3D carrier transport**: lateral (x–y) + vertical (z)
- Optimize **EQE, charge balance, recombination distribution, emission uniformity**

QLED-RLopt 面向科研应用，结合**强化学习算法**与**器件物理仿真**，用于自动探索和优化 QLED 器件结构，重点关注：
- 多层与微/纳结构（如 ZnO / QD / HTL 交替堆叠或平面图案化）
- 同时考虑横向 (x–y) 与纵向 (z) 的三维载流子输运行为
- 关键指标：外量子效率（EQE）、电荷注入平衡、复合区分布、发光均匀性等

This repository is designed to be:
- **Scientifically rigorous** – physics-informed reward and constraints
- **Modular & extensible** – plug in COMSOL/Lumerical/TCAD or surrogate models
- **Application-ready** for computational materials/device groups (e.g. SAM Lab)

设计理念：**不是一次性脚本，而是可插拔、可扩展的研究基础设施**。

---

## 2. Key Ideas | 核心思路

1. Treat QLED architecture search as a **sequential decision problem**.
2. Use an RL agent to propose:
   - Layer sequences (HTL / QD / ZnO / interlayers)
   - Micro/nano patterns (e.g., ZnO:QD fill factor, repeated stacks)
   - Thicknesses, doping, interface modifiers
3. Evaluate each design via:
   - Direct device simulations (COMSOL / Lumerical / other TCAD)
   - Or a learned **surrogate model** approximating those simulations
4. Optimize for:
   - High EQE
   - Strong electron–hole overlap
   - Smooth recombination profiles (no harsh hotspots)
   - Reasonable operating voltage & robustness


1. 将器件结构设计视作 **序列决策问题**。
2. RL agent 负责提出结构候选：
   - 层序（HTL / QD / ZnO / 缓冲层等）
   - ZnO–QD 平面占比、重复周期、多层发光区等几何参数
3. 通过仿真或代理模型评估：
   - EQE / 辐射与非辐射复合
   - 电子–空穴空间重叠
   - 电压、非均匀性、极端参数的惩罚项
4. 奖励函数物理约束，引导搜索落在 **可实现 + 有物理意义** 的结构空间。

---


## 3. Features | 功能特性

- 🔁 **RL Environment**
  - Encodes QLED structures into a compact parameter space
  - Interfaces with simulators or surrogate models

- 🤖 **Pluggable RL Agents**
  - Baseline DQN / policy-gradient implementations
  - Easy to swap/customize algorithms

- 📊 **Physics-Guided Reward**
  - Combines EQE, recombination uniformity, and penalty terms
  - Encourages physically meaningful, fabricable designs

- 🧪 **Surrogate Modeling (Optional)**
  - Train ML models on simulation data to accelerate exploration

- 📈 **Analysis & Visualization**
  - Jupyter notebooks for:
    - Design-performance landscapes
    - 2D/3D carrier & recombination maps
    - RL training curves


- 🔁 **RL 环境封装 / RL Environment**
  - 统一管理结构参数、设计采样与评估调用
- 🤖 **可插拔智能体 / Pluggable Agents**
  - 提供 DQN stub，支持替换为 PPO / A2C 等
- 🧪 **物理驱动奖励 / Physics-Guided Reward**
  - 同时考虑 EQE、载流子重叠、非辐射损失、工作电压等
- ⚡ **代理模型支持 / Surrogate Support**
  - 基于仿真数据训练 MLP / GNN，加速大规模搜索
- 📊 **可视化与分析 / Visualization**
  - Jupyter Notebooks 展示设计–性能关系与 RL 收敛过程

---



## 4. Repository Structure | 仓库结构

```text
QLED-RLopt/
├── README.md                    # 项目说明（本文件）
├── requirements.txt             # 依赖列表
├── qled_env/
│   ├── __init__.py
│   ├── parameter_space.py       # 器件结构参数编码与随机采样
│   ├── reward_function.py       # 物理约束奖励函数
│   ├── simulator_interface.py   # 统一调度 Mock / COMSOL / 代理模型
│   └── comsol_parser.py         # 解析 COMSOL 导出 CSV
├── agent/
│   ├── __init__.py
│   └── dqn_agent.py             # DQN 占位实现（可替换为真实RL算法）
├── surrogate_model/
│   ├── __init__.py
│   ├── train_surrogate.py       # 使用仿真数据训练代理模型
│   └── predict_performance.py   # 基于代理模型预测性能
├── data/
│   ├── generated_designs.csv    # 示例/占位设计与指标
│   └── simulated_results/       # 存放仿真或实验输出（CSV等）
├── scripts/
│   ├── run_optimization.py      # 主强化学习优化入口
│   └── simulate_design.py       # 单次结构评估（预留）
├── notebooks/
│   ├── 01_explore_parameter_space.ipynb
│   └── 02_visualize_rl_results.ipynb
├── tests/
│   ├── test_reward_logic.py     # 奖励函数单测
│   └── test_simulator_interface.py
└── LICENSE

```
---
## 5. Installation | 安装
git clone https://github.com/<your-username>/QLED-RLopt.git
cd QLED-RLopt
pip install -r requirements.txt



Python ≥ 3.9，默认依赖：numpy, pandas, scipy, matplotlib, torch, tqdm 等。

---

## 6. Quick Start | 快速开始
6.1 运行强化学习优化
python scripts/run_optimization.py --episodes 50


可选参数：

--use_surrogate 使用训练好的代理模型

--use_comsol 从 COMSOL CSV 读取真实仿真结果（开发者模式）

脚本将：

生成候选结构

使用 QLEDSimulator 计算指标

根据奖励函数更新 agent（当前为占位实现）

将数据记录在 data/ 下，便于可视化。
---

## 7. COMSOL / Surrogate Integration | 仿真与代理集成

使用 COMSOL / Lumerical 导出包含 x,y,z,n_electron,n_hole,R_rad,R_nrad 等字段的 CSV

放入 data/simulated_results/

在 design 中指定 comsol_csv 路径，并启用 --use_comsol

使用 surrogate_model/train_surrogate.py 基于高保真数据训练代理模型

该设计使本仓库可自然嵌入真实 QLED 仿真工作流。
---

## 8. Academic Use | 学术使用

如在论文、报告或申请材料中使用本框架，```

简要说明奖励设计与仿真边界条件

在合适位置引用或附上本仓库链接
---

## 9. License | 许可

使用 MIT License（默认），支持团队协作与二次开发。
---


