# 🛴 Segway — Pendule Inversé & Contrôle | Inverted Pendulum Control

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific-013243?style=for-the-badge&logo=numpy)
![SciPy](https://img.shields.io/badge/SciPy-LQR%20%7C%20RK4-8CAAE6?style=for-the-badge&logo=scipy)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Active-3fb950?style=for-the-badge)

**[🇫🇷 Français](#français) · [🇬🇧 English](#english)**

</div>

---

## 🇫🇷 Français <a name="français"></a>

### 📌 Description

Simulation physique complète d'un **Segway modélisé comme un pendule inversé sur roues**, avec implémentation et comparaison de trois lois de commande :

- **PID** — régulateur classique sur l'angle (démontre ses limites)
- **LQR** — retour d'état optimal par minimisation quadratique (Riccati)
- **LQG** — LQR + Filtre de Kalman pour conditions réelles avec bruit capteurs

Dashboard interactif Streamlit avec animation 2D du Segway, graphiques temps réel et tuning des gains en direct.

---

### 🎯 Objectif pédagogique

Démontrer **pourquoi le PID est insuffisant** pour un système multi-états instable, et prouver par simulation que le **LQR consomme 8× moins d'énergie** tout en stabilisant les 4 états simultanément.

| Contrôleur | Statut | Angle max | Énergie |
|:---:|:---:|:---:|:---:|
| PID | ❌ TOMBÉ | 46.6° | 368 J |
| LQR | ✅ STABLE | 8.0° | **45 J** |
| LQG | ✅ STABLE | 20.0° | 344 J |

> *Conditions : angle initial 8°, perturbation 20N à t=3s*

---

### 🧠 Modélisation physique

Le Segway est modélisé comme un **pendule inversé sur chariot** avec les équations de Newton-Euler (formulation standard Ogata) :

```
ẍ  = [F + m·sinθ·(L·θ̇² − g·cosθ) − b·ẋ] / (M + m·sin²θ)
θ̈  = [−cosθ·(F + m·L·θ̇²·sinθ − b·ẋ) + (M+m)·g·sinθ] / (L·(M + m·sin²θ))
```

**Paramètres physiques :**

| Paramètre | Valeur | Description |
|---|:---:|---|
| M | 2.0 kg | Masse châssis |
| m | 3.0 kg | Masse corps haut |
| L | 0.4 m | Hauteur centre de gravité |
| g | 9.81 m/s² | Gravité |
| b | 0.3 N·m·s | Frottement visqueux |
| dt | 0.005 s | Pas de temps (5ms — 200 Hz) |

**Intégration numérique :** Runge-Kutta ordre 4 (RK4)

**Linéarisation** autour de θ=0 → représentation d'état :
```
ẋ = A·x + B·u     x = [x, ẋ, θ, θ̇]ᵀ     u = F (force moteur)
```

---

### 🔧 Lois de commande implémentées

#### PID — Régulateur classique
```
F = Kp·θ + Ki·∫θ dt + Kd·θ̇
```
Contrôle l'angle seul. Ignore position, vitesse et vitesse angulaire → **structurellement insuffisant**.

#### LQR — Linear Quadratic Regulator
```
F = −K·[x, ẋ, θ, θ̇]ᵀ

K = R⁻¹·Bᵀ·P     (P = solution équation de Riccati)
```
Minimise `J = ∫(xᵀQx + F²R)dt` — **optimal en énergie**, contrôle les 4 états.

Paramètres utilisés :
```
Q = diag([1, 1, 100, 10])   → angle prioritaire (×100)
R = 0.1                      → compromis performance / énergie
```

#### LQG — LQR + Filtre de Kalman
```
Predict : x̂⁻ = Ad·x̂ + Bd·F        P⁻ = Ad·P·Ad' + Q_noise
Update  : K_kalman = P⁻·C'·(C·P⁻·C' + R_noise)⁻¹
          x̂ = x̂⁻ + K_kalman·(y − C·x̂⁻)
```
Estime l'état réel à partir de mesures bruitées. Erreur moyenne : **1.258 mrad** sur θ.

---

### 🏗️ Architecture du projet

```
Segway_pendule_inverse_control/
├── src/
│   ├── __init__.py
│   ├── segway_model.py       # Équations physiques RK4 + perturbations
│   ├── controllers.py        # PID (anti-windup) + LQR (Riccati)
│   └── kalman_filter.py      # Filtre de Kalman (predict + update)
├── app.py                    # Dashboard Streamlit 3 onglets
├── test_model.py             # Tests des 4 scénarios
├── requirements.txt
└── README.md
```

---

### 📊 Dashboard Streamlit — 3 onglets

**Onglet 1 — Simulation**
- Choix contrôleur : LQR / LQG / PID
- Segway animé en 2D (tige verte→orange→rouge selon l'angle)
- Flèche de force moteur en temps réel
- Graphiques : angle θ, position x, force F
- Injection de perturbation (poussée pilote)

**Onglet 2 — Comparaison PID vs LQR vs LQG**
- Simulation simultanée des 3 contrôleurs
- Courbes superposées angle et force
- Tableau métriques : statut, angle max, énergie

**Onglet 3 — Tuning LQR**
- Sliders Q et R en temps réel
- Gains K calculés automatiquement
- Pôles en boucle fermée avec indicateur stabilité 🟢/🔴

---

### 🚀 Installation

```bash
# 1. Cloner le repo
git clone https://github.com/aliouha/Segway_pendule_inverse_control.git
cd Segway_pendule_inverse_control

# 2. Créer l'environnement virtuel
python -m venv venv
.\venv\Scripts\activate          # Windows
source venv/bin/activate         # Linux / Mac

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Tester le modèle
python test_model.py

# 5. Lancer le dashboard
streamlit run app.py
```

---

### 📋 Résultats test_model.py

```
==================================================
SEGWAY — TEST CONTRÔLEURS
==================================================
[1] PID sur angle seul
    ❌ Tombé à t=0.24s
    Position x = -0.24m (dérive !)

[2] LQR — retour d'état complet
    ✅ Stable après 10.0s
    Angle final  : 0.000149 rad ≈ 0
    Position x   : 0.0123 m

[3] LQR avec poussée pilote à t=3s (20N)
    ✅ Stable malgré perturbation !
    Angle final : -0.000235 rad

[4] LQG — LQR + Filtre de Kalman
    ✅ LQG stable après 10.0s avec bruit capteurs
    Erreur estimation theta : 1.258 mrad
    Angle final : 0.1005 mrad ≈ 0
==================================================
```

---

### 🌍 Applications réelles

| Application | Contrôleur réel |
|---|---|
| 🛴 Segway / trottinettes auto-équilibrées | PID cascade → LQG |
| 🚀 SpaceX Falcon 9 atterrissage vertical | LQR + MPC + Kalman |
| 🤖 Boston Dynamics Atlas (marche) | MPC haute fréquence 500Hz |
| 🚁 Drones quadrotors (stabilisation) | PID cascade → LQR |
| 🏗️ Grues industrielles (anti-oscillation) | PID + Input Shaping |

---

### 📦 Stack technique

| Outil | Usage |
|---|---|
| Python 3.10+ | Langage principal |
| NumPy | Calcul matriciel, algèbre linéaire |
| SciPy | Équation de Riccati, intégration RK4 |
| Matplotlib | Graphiques et animation 2D Segway |
| Streamlit | Dashboard interactif |

---

---

## 🇬🇧 English <a name="english"></a>

### 📌 Description

Full physical simulation of a **Segway modeled as a wheeled inverted pendulum**, with implementation and comparison of three control laws:

- **PID** — classical angle controller (demonstrates its limitations)
- **LQR** — optimal state feedback via quadratic minimization (Riccati)
- **LQG** — LQR + Kalman Filter for real-world noisy sensor conditions

Interactive Streamlit dashboard with 2D Segway animation, real-time plots, and live gain tuning.

---

### 🎯 Educational Goal

Demonstrate **why PID fails** on a multi-state unstable system, and prove through simulation that **LQR uses 8× less energy** while stabilizing all 4 states simultaneously.

| Controller | Status | Max Angle | Energy |
|:---:|:---:|:---:|:---:|
| PID | ❌ FALLEN | 46.6° | 368 J |
| LQR | ✅ STABLE | 8.0° | **45 J** |
| LQG | ✅ STABLE | 20.0° | 344 J |

> *Test: initial angle 8°, 20N disturbance at t=3s*

---

### 🧠 Physical Modeling

The Segway is modeled as an **inverted pendulum on a cart** using Newton-Euler equations (Ogata standard formulation):

```
ẍ  = [F + m·sinθ·(L·θ̇² − g·cosθ) − b·ẋ] / (M + m·sin²θ)
θ̈  = [−cosθ·(F + m·L·θ̇²·sinθ − b·ẋ) + (M+m)·g·sinθ] / (L·(M + m·sin²θ))
```

**Numerical integration:** Runge-Kutta 4th order (RK4)

**Linearization** around θ=0 → state-space:
```
ẋ = A·x + B·u     x = [x, ẋ, θ, θ̇]ᵀ     u = F (motor force)
```

---

### 🔧 Control Laws

**PID** — Controls angle only. Structurally insufficient for this system.

**LQR** — Minimizes `J = ∫(xᵀQx + F²R)dt`. Optimal energy, controls all 4 states.

**LQG** — LQR + Kalman Filter. Estimates true state from noisy sensor measurements. Mean estimation error: **1.258 mrad**.

---

### 🚀 Installation

```bash
git clone https://github.com/aliouha/Segway_pendule_inverse_control.git
cd Segway_pendule_inverse_control
python -m venv venv && .\venv\Scripts\activate
pip install -r requirements.txt
python test_model.py    # Run tests
streamlit run app.py    # Launch dashboard
```

---

### 🌍 Real-World Applications

| Application | Controller |
|---|---|
| 🛴 Self-balancing scooters | Cascaded PID → LQG |
| 🚀 SpaceX Falcon 9 landing | LQR + MPC + Kalman |
| 🤖 Boston Dynamics Atlas | High-frequency MPC (500Hz) |
| 🚁 Quadrotor drones | Cascaded PID → LQR |

---

### 🔑 Key Findings

**PID is structurally insufficient** — only controls θ, ignores x, ẋ, θ̇ → position drifts → falls at 0.24s.

**LQR is energy-optimal** — 45J vs 368J for PID, **8× more efficient**, stabilizes all 4 states.

**LQG is essential for real deployment** — Kalman filter handles IMU and encoder noise, mean error 1.258 mrad.

---

<div align="center">

**Auteur | Author : Aliou Harber**

[![Portfolio](https://img.shields.io/badge/Portfolio-aliouha.github.io-blue?style=flat-square)](https://aliouha.github.io)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-aliou--harber-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/aliou-harber)
[![GitHub](https://img.shields.io/badge/GitHub-aliouha-181717?style=flat-square&logo=github)](https://github.com/aliouha)

*🛴 Segway Inverted Pendulum Control — FST Settat · Université Hassan 1er*

</div>