"""
segway_model.py
===============
Modèle physique d'un Segway (pendule inversé sur roues)
Intégration numérique par Runge-Kutta ordre 4 (RK4)
Formulation standard textbook (Ogata)

Auteur : Aliou Harber
"""

import numpy as np

# ══════════════════════════════════════════════════════
# PARAMÈTRES PHYSIQUES
# ══════════════════════════════════════════════════════
PARAMS = {
    "M"  : 2.0,    # kg   — masse du châssis
    "m"  : 3.0,    # kg   — masse du pendule (corps haut)
    "L"  : 0.4,    # m    — longueur tige
    "g"  : 9.81,   # m/s²
    "b"  : 0.3,    # N·m·s — frottement
    "dt" : 0.005,  # s    — pas de temps (5ms)
}


# ══════════════════════════════════════════════════════
# ÉQUATIONS DU MOUVEMENT (non linéaires — formulation standard)
# ══════════════════════════════════════════════════════
def equations(state, F, p):
    """
    Équations du pendule inversé sur chariot.
    theta = 0 → pendule vertical haut (équilibre instable)
    theta > 0 → penche à droite

    state = [x, x_dot, theta, theta_dot]
    F     = force moteur (N)
    """
    M, m, L, g, b = p["M"], p["m"], p["L"], p["g"], p["b"]
    x, xd, th, thd = state

    s, c  = np.sin(th), np.cos(th)
    denom = M + m * s**2

    # Accélération chariot
    xdd = (F + m * s * (L * thd**2 - g * c) - b * xd) / denom

    # Accélération angulaire
    thdd = (-c * (F + m * L * thd**2 * s - b * xd) \
            + (M + m) * g * s) / (L * denom)

    return np.array([xd, xdd, thd, thdd])


# ══════════════════════════════════════════════════════
# INTÉGRATION RUNGE-KUTTA ORDRE 4
# ══════════════════════════════════════════════════════
def rk4_step(state, F, p):
    dt = p["dt"]
    k1 = equations(state,            F, p)
    k2 = equations(state + dt/2*k1,  F, p)
    k3 = equations(state + dt/2*k2,  F, p)
    k4 = equations(state + dt*k3,    F, p)
    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


# ══════════════════════════════════════════════════════
# MATRICES LINÉARISÉES (pour LQR)
# ══════════════════════════════════════════════════════
def get_linear_matrices(p):
    """
    Système linéarisé autour de theta=0.
    ẋ = A·x + B·u
    """
    M, m, L, g, b = p["M"], p["m"], p["L"], p["g"], p["b"]
    d = M + m  # dénominateur à theta=0

    A = np.array([
        [0,  1,              0,           0],
        [0,  -b/d,           -m*g/d,      0],
        [0,  0,              0,           1],
        [0,  b/(L*d),        (M+m)*g/(L*d), 0],
    ])

    B = np.array([
        [0],
        [1/d],
        [0],
        [-1/(L*d)],
    ])

    return A, B


# ══════════════════════════════════════════════════════
# PERTURBATIONS
# ══════════════════════════════════════════════════════
class Disturbance:

    @staticmethod
    def rider_push(t, t_push=5.0, magnitude=15.0, duration=0.1):
        if t_push <= t <= t_push + duration:
            return magnitude
        return 0.0

    @staticmethod
    def sensor_noise(std_theta=0.003, std_x=0.005):
        return np.random.normal(0, std_theta), np.random.normal(0, std_x)


# ══════════════════════════════════════════════════════
# SIMULATEUR PRINCIPAL
# ══════════════════════════════════════════════════════
class SegwaySimulator:

    def __init__(self, params=None, theta_init=0.1):
        self.params = params or PARAMS
        self.reset(theta_init)

    def reset(self, theta_init=0.1):
        self.state = np.array([0.0, 0.0, theta_init, 0.0])
        self.t     = 0.0
        self.history = {
            "t": [], "x": [], "x_dot": [],
            "theta": [], "theta_dot": [], "F": []
        }

    def step(self, F, disturbance=0.0):
        F_total    = np.clip(F + disturbance, -100, 100)
        self.state = rk4_step(self.state, F_total, self.params)
        self.t    += self.params["dt"]

        x, xd, th, thd = self.state
        self.history["t"].append(round(self.t, 4))
        self.history["x"].append(x)
        self.history["x_dot"].append(xd)
        self.history["theta"].append(np.degrees(th))
        self.history["theta_dot"].append(thd)
        self.history["F"].append(F_total)
        return self.state

    def get_measured_state(self, add_noise=True):
        state = self.state.copy()
        if add_noise:
            n_th, n_x = Disturbance.sensor_noise()
            state[0] += n_x
            state[2] += n_th
        return state

    @property
    def is_fallen(self):
        return abs(self.state[2]) > np.radians(45)