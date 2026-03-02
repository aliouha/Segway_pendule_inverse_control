"""
controllers.py
==============
PID et LQR pour le Segway

Auteur : Aliou Harber
"""

import numpy as np
from scipy.linalg import solve_continuous_are


class PIDController:

    def __init__(self, Kp=30.0, Ki=2.0, Kd=8.0,
                 setpoint=0.0, F_max=100.0):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint  = setpoint
        self.F_max     = F_max
        self._integral = 0.0
        self._prev_err = 0.0
        self._first    = True

    def reset(self):
        self._integral = 0.0
        self._prev_err = 0.0
        self._first    = True

    def compute(self, theta, dt):
        e  = self.setpoint - theta
        P  = self.Kp * e
        self._integral += e * dt
        I  = self.Ki * self._integral
        D  = 0.0 if self._first else self.Kd * (e - self._prev_err) / dt
        self._first    = False
        self._prev_err = e
        F     = P + I + D
        F_sat = np.clip(F, -self.F_max, self.F_max)
        # anti-windup
        if F != F_sat:
            self._integral -= e * dt
        return F_sat


class LQRController:

    def __init__(self, A, B, Q=None, R=None, F_max=100.0):
        self.F_max = F_max
        if Q is None:
            Q = np.diag([1.0, 1.0, 100.0, 10.0])
        if R is None:
            R = np.array([[0.1]])
        self.K = self._compute_K(A, B, Q, R)
        print(f"LQR K = {self.K.round(3)}")

    def _compute_K(self, A, B, Q, R):
        P = solve_continuous_are(A, B, Q, R)
        return np.linalg.inv(R) @ B.T @ P

    def compute(self, state):
        F = np.dot(-self.K.flatten(), state)
        return np.clip(F, -self.F_max, self.F_max)

    def tune(self, Q_diag, R_val, A, B):
        self.K = self._compute_K(A, B, np.diag(Q_diag), np.array([[R_val]]))


def get_controller(name, A=None, B=None, **kwargs):
    if name == "pid":
        return PIDController(**kwargs)
    elif name == "lqr":
        return LQRController(A, B, **kwargs)
    else:
        raise ValueError(f"Inconnu : {name}")