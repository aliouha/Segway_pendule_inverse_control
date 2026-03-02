import numpy as np
import sys
sys.path.append('src')

from segway_model import SegwaySimulator, get_linear_matrices, PARAMS
from controllers  import get_controller

print("=" * 50)
print("SEGWAY — TEST CONTRÔLEURS")
print("=" * 50)

# ── Test PID ──────────────────────────────────────
print("\n[1] PID sur angle seul")
print("    → contrôle seulement theta, ignore position x")
sim = SegwaySimulator(theta_init=0.1)
pid = get_controller("pid", Kp=30, Ki=2, Kd=8)

for i in range(2000):
    state = sim.get_measured_state(add_noise=False)
    F     = pid.compute(state[2], PARAMS["dt"])
    sim.step(F)
    if sim.is_fallen:
        print(f"    ❌ Tombé à t={sim.t:.2f}s")
        print(f"    Position x = {sim.state[0]:.2f}m (dérive !)")
        break
else:
    print(f"    ✅ Stable {sim.t:.1f}s")

# ── Test LQR ──────────────────────────────────────
print("\n[2] LQR — retour d'état complet")
print("    → contrôle x, x_dot, theta, theta_dot simultanément")
sim.reset(theta_init=0.1)
A, B = get_linear_matrices(PARAMS)
lqr  = get_controller("lqr", A=A, B=B)

for i in range(2000):
    state = sim.get_measured_state(add_noise=False)
    F     = lqr.compute(state)
    sim.step(F)
    if sim.is_fallen:
        print(f"    ❌ Tombé à t={sim.t:.2f}s"); break
else:
    print(f"    ✅ Stable après {sim.t:.1f}s")
    print(f"    Angle final  : {sim.state[2]:.6f} rad ≈ 0")
    print(f"    Position x   : {sim.state[0]:.4f} m")

# ── Test LQR avec perturbation ─────────────────────
print("\n[3] LQR avec poussée pilote à t=3s")
sim.reset(theta_init=0.05)
A, B = get_linear_matrices(PARAMS)
lqr2 = get_controller("lqr", A=A, B=B)

from segway_model import Disturbance
for i in range(2000):
    state = sim.get_measured_state(add_noise=False)
    F     = lqr2.compute(state)
    dist  = Disturbance.rider_push(sim.t, t_push=3.0, magnitude=20.0)
    sim.step(F, disturbance=dist)
    if sim.is_fallen:
        print(f"    ❌ Tombé à t={sim.t:.2f}s"); break
else:
    print(f"    ✅ Stable malgré perturbation !")
    print(f"    Angle final : {sim.state[2]:.6f} rad")

print("\n" + "=" * 50)
print("CONCLUSION :")
print("  PID    → ❌ insuffisant pour pendule inversé")
print("  LQR    → ✅ optimal, stabilise les 4 états")
print("  LQR+perturbation → ✅ robuste")
print("=" * 50)

# ── Test Kalman + LQR ─────────────────────────────
print("\n[4] LQG — LQR + Filtre de Kalman (bruit capteurs)")
from segway_model import SegwaySimulator, get_linear_matrices, PARAMS
from controllers  import get_controller
from kalman_filter import KalmanFilter

sim.reset(theta_init=0.1)
A, B = get_linear_matrices(PARAMS)
lqr_k = get_controller("lqr", A=A, B=B)
kf    = KalmanFilter(A, B, PARAMS["dt"])
kf.reset(x0=np.array([0.0, 0.0, 0.1, 0.0]))

errors = []
F = 0.0

for i in range(2000):
    # Mesure bruitée (ce que lirait vraiment un capteur)
    noisy = sim.get_measured_state(add_noise=True)
    measurement = np.array([noisy[0], noisy[2]])  # [x, theta]

    # Kalman estime l'état réel
    x_est = kf.step(F, measurement)

    # LQR utilise l'état estimé (pas le bruité)
    F = lqr_k.compute(x_est)

    # Simulation avance
    sim.step(F)

    # Erreur estimation
    err = abs(x_est[2] - sim.state[2])
    errors.append(err)

    if sim.is_fallen:
        print(f"    ❌ Tombé à t={sim.t:.2f}s"); break
else:
    print(f"    ✅ LQG stable après {sim.t:.1f}s avec bruit capteurs")
    print(f"    Erreur estimation theta (moy) : {np.mean(errors)*1000:.3f} mrad")
    print(f"    Angle final  : {sim.state[2]*1000:.4f} mrad ≈ 0")