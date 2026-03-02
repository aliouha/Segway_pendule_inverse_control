"""
kalman_filter.py
================
Filtre de Kalman pour le Segway.

Problème réel : les capteurs (IMU, encodeurs) sont BRUITÉS.
Le Kalman estime l'état réel à partir des mesures bruitées.

Auteur : Aliou Harber
"""

import numpy as np


class KalmanFilter:
    """
    Filtre de Kalman Linéaire.

    Cycle à chaque pas de temps :
      1. PREDICT  → prédit le nouvel état via le modèle
      2. UPDATE   → corrige avec la mesure capteur

    État estimé : x̂ = [x, x_dot, theta, theta_dot]
    Mesures     : y  = [x, theta]  (encodeur + IMU)
    """

    def __init__(self, A, B, dt,
                 Q_noise=None, R_noise=None, P0=None):
        """
        A, B    : matrices système linéarisé
        dt      : pas de temps
        Q_noise : bruit processus (incertitude modèle)
        R_noise : bruit mesure   (incertitude capteurs)
        P0      : covariance initiale
        """
        self.dt = dt
        n = A.shape[0]  # 4 états

        # Discrétisation A → Ad = I + A*dt
        self.Ad = np.eye(n) + A * dt
        self.Bd = B * dt

        # Matrice d'observation : on mesure x et theta
        self.C = np.array([
            [1, 0, 0, 0],   # mesure x
            [0, 0, 1, 0],   # mesure theta
        ])

        # Bruit processus Q — incertitude sur le modèle
        if Q_noise is None:
            Q_noise = np.diag([0.001, 0.01, 0.001, 0.01])

        # Bruit mesure R — incertitude capteurs
        if R_noise is None:
            R_noise = np.diag([0.05, 0.005])
            # x bruité (encodeur)  → 0.05
            # theta bruité (IMU)   → 0.005 (plus précis)

        self.Q = Q_noise
        self.R = R_noise

        # Covariance initiale P
        self.P = P0 if P0 is not None else np.eye(n) * 0.1

        # État estimé initial
        self.x_hat = np.zeros(n)

    def reset(self, x0=None):
        """Remet le filtre à zéro."""
        n = self.Ad.shape[0]
        self.x_hat = x0 if x0 is not None else np.zeros(n)
        self.P     = np.eye(n) * 0.1

    def predict(self, F):
        """
        Étape PREDICT — propagation temporelle.
        Prédit où sera le système au prochain pas.

        x̂⁻ = Ad·x̂ + Bd·F
        P⁻  = Ad·P·Ad' + Q
        """
        self.x_hat = self.Ad @ self.x_hat + self.Bd.flatten() * F
        self.P     = self.Ad @ self.P @ self.Ad.T + self.Q

    def update(self, measurement):
        """
        Étape UPDATE — correction par la mesure.

        K   = P⁻·C'·(C·P⁻·C' + R)⁻¹   ← gain de Kalman
        x̂   = x̂⁻ + K·(y − C·x̂⁻)       ← correction
        P   = (I − K·C)·P⁻              ← mise à jour covariance

        measurement : [x_mesuré, theta_mesuré]
        """
        # Innovation (écart entre mesure et prédiction)
        y_pred    = self.C @ self.x_hat
        innovation = measurement - y_pred

        # Gain de Kalman
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)

        # Correction état et covariance
        self.x_hat = self.x_hat + K @ innovation
        self.P     = (np.eye(len(self.x_hat)) - K @ self.C) @ self.P

        return self.x_hat.copy()

    def step(self, F, measurement):
        """
        Predict + Update en une seule appel.
        Retourne l'état estimé x̂.
        """
        self.predict(F)
        return self.update(measurement)

    @property
    def state_estimate(self):
        return self.x_hat.copy()