import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
sys.path.append("src")

from segway_model import SegwaySimulator, get_linear_matrices, PARAMS, Disturbance
from controllers  import get_controller
from kalman_filter import KalmanFilter
from scipy.linalg import solve_continuous_are

st.set_page_config(page_title="Segway — Pendule Inversé", page_icon="🛴", layout="wide")

st.markdown("""
<style>
  .metric-card {
    background: #1c2128; border: 1px solid #30363d;
    border-radius: 10px; padding: 16px; text-align: center;
  }
  .metric-val { font-size: 1.8rem; font-weight: 800; }
  .metric-lbl { font-size: 0.8rem; color: #8b949e; margin-top: 4px; }
  .stable   { color: #3fb950; }
  .unstable { color: #f85149; }
  .warning  { color: #d29922; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════
def run_simulation(controller_name, theta_init, t_push, push_mag,
                   add_noise, Q_diag, R_val, duration=10.0):
    p    = PARAMS.copy()
    sim  = SegwaySimulator(params=p, theta_init=theta_init)
    A, B = get_linear_matrices(p)

    if controller_name == "LQG":
        ctrl = get_controller("lqr", A=A, B=B, Q=np.diag(Q_diag), R=np.array([[R_val]]))
        kf   = KalmanFilter(A, B, p["dt"])
        kf.reset(x0=np.array([0.0, 0.0, theta_init, 0.0]))
    elif controller_name == "LQR":
        ctrl = get_controller("lqr", A=A, B=B, Q=np.diag(Q_diag), R=np.array([[R_val]]))
        kf   = None
    else:
        ctrl = get_controller("pid", Kp=30, Ki=2, Kd=8)
        kf   = None

    n_steps        = int(duration / p["dt"])
    F              = 0.0
    fallen_at      = None
    theta_est_hist = []

    for i in range(n_steps):
        t    = sim.t
        dist = Disturbance.rider_push(t, t_push=t_push, magnitude=push_mag, duration=0.15)

        if controller_name == "LQG":
            noisy = sim.get_measured_state(add_noise=True)
            meas  = np.array([noisy[0], noisy[2]])
            x_est = kf.step(F, meas)
            F     = ctrl.compute(x_est)
            theta_est_hist.append(np.degrees(x_est[2]))
        elif controller_name == "LQR":
            state = sim.get_measured_state(add_noise=add_noise)
            F     = ctrl.compute(state)
            theta_est_hist.append(np.degrees(state[2]))
        else:
            state = sim.get_measured_state(add_noise=add_noise)
            F     = ctrl.compute(state[2], p["dt"])
            theta_est_hist.append(np.degrees(state[2]))

        sim.step(F, disturbance=dist)

        if sim.is_fallen and fallen_at is None:
            fallen_at = sim.t
            break

    hist               = sim.history
    hist["theta_est"]  = theta_est_hist
    hist["fallen_at"]  = fallen_at
    return hist


# ══════════════════════════════════════════════════════
# DRAW SEGWAY
# ══════════════════════════════════════════════════════
def draw_segway(theta_deg, x_pos, F, ax):
    ax.set_facecolor("#0e1117")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")

    theta   = np.radians(theta_deg)
    xc      = np.clip(x_pos, -2.5, 2.5)
    wheel_r = 0.17

    ax.axhline(y=0, color="#30363d", linewidth=2, zorder=0)
    ax.fill_between([-3, 3], -0.5, 0, color="#161b22", zorder=0)

    for dx in [-0.2, 0.2]:
        ax.add_patch(plt.Circle((xc+dx, wheel_r), wheel_r, color="#238636", fill=True, zorder=2))
        ax.add_patch(plt.Circle((xc+dx, wheel_r), wheel_r*0.4, color="#0e1117", zorder=3))

    ax.add_patch(patches.FancyBboxPatch(
        (xc-0.25, wheel_r*2-0.05), 0.5, 0.2,
        boxstyle="round,pad=0.02",
        facecolor="#1f6feb", edgecolor="#58a6ff", linewidth=2, zorder=3))

    L         = 0.8
    x_top     = xc + L * np.sin(theta)
    y_bot     = wheel_r*2 + 0.15
    y_top     = y_bot + L * np.cos(theta)
    color_rod = "#f85149" if abs(theta_deg) > 20 else "#d29922" if abs(theta_deg) > 10 else "#3fb950"
    ax.plot([xc, x_top], [y_bot, y_top], color=color_rod, linewidth=5,
            solid_capstyle="round", zorder=4)
    ax.add_patch(plt.Circle((x_top, y_top), 0.12, color="#58a6ff", zorder=5))

    if abs(F) > 0.5:
        arrow_dir = 1 if F > 0 else -1
        arrow_len = min(abs(F)/30, 0.6)
        ax.annotate("",
            xy=(xc + arrow_dir*arrow_len, wheel_r), xytext=(xc, wheel_r),
            arrowprops=dict(arrowstyle="->", color="#f0883e", lw=2.5, mutation_scale=15))

    status = "🟢 STABLE" if abs(theta_deg) < 10 else "🟡 LIMITE" if abs(theta_deg) < 25 else "🔴 TOMBÉ"
    ax.set_title(f"θ = {theta_deg:.2f}°  |  x = {x_pos:.3f}m  |  F = {F:.1f}N  |  {status}",
                 color="white", fontsize=10, pad=8)


# ══════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════
st.markdown("# 🛴 Segway — Contrôle du Pendule Inversé")
st.markdown("**Simulation physique** · PID · LQR · LQG (Kalman) · par **Aliou Harber**")
st.divider()

tab1, tab2, tab3 = st.tabs(["🎮 Simulation", "📊 Comparaison PID vs LQR vs LQG", "🔧 Tuning LQR"])

# ════════════════════════════════════════════════════════
# TAB 1 — SIMULATION
# ════════════════════════════════════════════════════════
with tab1:
    col_side, col_main = st.columns([1, 2.5])

    with col_side:
        st.markdown("### ⚙️ Paramètres")
        controller = st.selectbox("Contrôleur", ["LQR", "LQG (Kalman)", "PID"])
        ctrl_name  = controller.split()[0]
        theta_init = st.slider("Angle initial θ₀ (°)", 1.0, 30.0, 10.0, 0.5)
        duration   = st.slider("Durée (s)", 5.0, 20.0, 10.0, 1.0)
        add_noise  = st.checkbox("Bruit capteurs", value=True)

        st.markdown("---")
        st.markdown("### 💥 Perturbation")
        t_push   = st.slider("Instant poussée (s)", 1.0, 8.0, 4.0, 0.5)
        push_mag = st.slider("Intensité (N)", 0.0, 50.0, 15.0, 1.0)

        st.markdown("---")
        st.markdown("### 🎛️ Gains LQR")
        q_angle = st.slider("Q[θ]", 10, 500, 100, 10)
        q_pos   = st.slider("Q[x]",  1,  50,   5,  1)
        r_val   = st.slider("R",  0.01, 1.0, 0.1, 0.01)
        Q_diag  = [float(q_pos), 1.0, float(q_angle), 10.0]

        if st.button("▶️ Lancer la simulation", use_container_width=True):
            with st.spinner("Calcul..."):
                hist = run_simulation(ctrl_name, np.radians(theta_init),
                                      t_push, push_mag, add_noise,
                                      Q_diag, r_val, duration)
            st.session_state["hist"]     = hist
            st.session_state["duration"] = duration
            st.session_state["t_push"]   = t_push
            st.session_state["push_mag"] = push_mag

    with col_main:
        if "hist" in st.session_state:
            hist     = st.session_state["hist"]
            t_push_s = st.session_state["t_push"]
            push_s   = st.session_state["push_mag"]
            fallen   = hist["fallen_at"]
            t_arr    = np.array(hist["t"])
            th_arr   = np.array(hist["theta"])
            x_arr    = np.array(hist["x"])
            F_arr    = np.array(hist["F"])
            th_est   = np.array(hist.get("theta_est", []))
            n        = len(t_arr)

            # Métriques
            c1, c2, c3, c4 = st.columns(4)
            status_txt = "TOMBÉ ❌" if fallen else "STABLE ✅"
            status_cls = "unstable" if fallen else "stable"
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-val {status_cls}">{status_txt}</div><div class="metric-lbl">Statut</div></div>', unsafe_allow_html=True)
            with c2:
                t_stab = f"{fallen:.2f}s" if fallen else f"{st.session_state['duration']:.0f}s"
                st.markdown(f'<div class="metric-card"><div class="metric-val warning">{t_stab}</div><div class="metric-lbl">Durée stable</div></div>', unsafe_allow_html=True)
            with c3:
                max_th = f"{np.max(np.abs(th_arr)):.1f}°"
                st.markdown(f'<div class="metric-card"><div class="metric-val">{max_th}</div><div class="metric-lbl">Angle max</div></div>', unsafe_allow_html=True)
            with c4:
                energy = f"{np.sum(F_arr**2)*PARAMS['dt']:.1f} J"
                st.markdown(f'<div class="metric-card"><div class="metric-val">{energy}</div><div class="metric-lbl">Énergie</div></div>', unsafe_allow_html=True)

            st.markdown("---")

            # Segway animé
            frame_idx = st.slider("🎞️ Frame animation", 0, n-1,
                                  st.session_state.get("frame", 0), key="frame_slider")
            st.session_state["frame"] = frame_idx

            fig_seg, ax_seg = plt.subplots(figsize=(8, 3.5), facecolor="#0e1117")
            draw_segway(th_arr[frame_idx], x_arr[frame_idx], F_arr[frame_idx], ax_seg)
            st.pyplot(fig_seg, use_container_width=True)
            plt.close(fig_seg)

            # Graphiques
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), facecolor="#0e1117")
            fig.subplots_adjust(hspace=0.4)
            plots = [
                (th_arr, th_est if len(th_est) > 0 else None, "Angle θ (°)",    "#3fb950", "#f0883e", "Réel", "Estimé"),
                (x_arr,  None,                                 "Position x (m)", "#58a6ff", None,      "x",    None),
                (F_arr,  None,                                 "Force (N)",      "#d29922", None,      "F",    None),
            ]
            for ax, (y1, y2, ylabel, c1c, c2c, l1, l2) in zip(axes, plots):
                ax.set_facecolor("#161b22")
                ax.plot(t_arr[:len(y1)], y1, color=c1c, lw=1.5, label=l1)
                if y2 is not None:
                    ax.plot(t_arr[:len(y2)], y2, color=c2c, lw=1, linestyle="--", alpha=0.7, label=l2)
                ax.axhline(0, color="#30363d", lw=0.8, linestyle="--")
                if push_s > 0:
                    ax.axvline(t_push_s, color="#f85149", lw=1, linestyle=":", alpha=0.7, label="Perturbation")
                ax.set_ylabel(ylabel, color="white", fontsize=9)
                ax.tick_params(colors="white", labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color("#30363d")
                ax.legend(fontsize=8, loc="upper right", facecolor="#1c2128", labelcolor="white")
                ax.grid(True, alpha=0.15)
            axes[-1].set_xlabel("Temps (s)", color="white", fontsize=9)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("👈 Configure et clique **▶️ Lancer la simulation**")


# ════════════════════════════════════════════════════════
# TAB 2 — COMPARAISON
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Comparaison PID vs LQR vs LQG")
    cc1, cc2 = st.columns(2)
    with cc1:
        th0_cmp  = st.slider("Angle initial (°)", 1.0, 20.0, 8.0,  key="cmp_th")
        push_cmp = st.slider("Perturbation (N)",  0.0, 40.0, 20.0, key="cmp_push")
    with cc2:
        # ← FIX : clé widget différente de la clé session_state
        tp_cmp = st.slider("Instant perturbation (s)", 1.0, 6.0, 3.0, key="cmp_tp_slider")

    if st.button("▶️ Comparer les 3 contrôleurs", use_container_width=True):
        Q_d = [1.0, 1.0, 100.0, 10.0]
        th0 = np.radians(th0_cmp)
        with st.spinner("Simulation des 3 contrôleurs..."):
            h_pid = run_simulation("PID", th0, tp_cmp, push_cmp, False, Q_d, 0.1, 10.0)
            h_lqr = run_simulation("LQR", th0, tp_cmp, push_cmp, False, Q_d, 0.1, 10.0)
            h_lqg = run_simulation("LQG", th0, tp_cmp, push_cmp, True,  Q_d, 0.1, 10.0)
        # ← FIX : clé session_state différente du widget
        st.session_state["cmp"]        = {"PID": h_pid, "LQR": h_lqr, "LQG": h_lqg}
        st.session_state["cmp_tp_val"] = tp_cmp

    if "cmp" in st.session_state:
        hists  = st.session_state["cmp"]
        # ← FIX : lit depuis la bonne clé
        tp_s   = st.session_state.get("cmp_tp_val", 3.0)
        colors = {"PID": "#f85149", "LQR": "#3fb950", "LQG": "#58a6ff"}

        fig2, axes2 = plt.subplots(2, 1, figsize=(11, 7), facecolor="#0e1117")
        fig2.subplots_adjust(hspace=0.4)
        for name, hist in hists.items():
            t  = np.array(hist["t"])
            th = np.array(hist["theta"])
            F  = np.array(hist["F"])
            for ax, y in zip(axes2, [th, F]):
                ax.set_facecolor("#161b22")
                ax.plot(t[:len(y)], y, color=colors[name], lw=1.8, label=name)

        for ax, lbl in zip(axes2, ["Angle θ (°)", "Force (N)"]):
            ax.axhline(0, color="#30363d", lw=0.8, linestyle="--")
            ax.axvline(tp_s, color="#d29922", lw=1.2, linestyle=":", label="Perturbation")
            ax.set_ylabel(lbl, color="white", fontsize=10)
            ax.tick_params(colors="white", labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#30363d")
            ax.legend(fontsize=9, facecolor="#1c2128", labelcolor="white")
            ax.grid(True, alpha=0.15)
        axes2[-1].set_xlabel("Temps (s)", color="white", fontsize=10)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        st.markdown("### 📋 Métriques comparatives")
        mc1, mc2, mc3 = st.columns(3)
        for col, (name, hist) in zip([mc1, mc2, mc3], hists.items()):
            fallen = hist["fallen_at"]
            th_arr = np.array(hist["theta"])
            F_arr  = np.array(hist["F"])
            status = "❌ TOMBÉ" if fallen else "✅ STABLE"
            energy = f"{np.sum(F_arr**2)*PARAMS['dt']:.0f} J"
            max_th = f"{np.max(np.abs(th_arr)):.1f}°"
            color  = "#f85149" if fallen else "#3fb950"
            with col:
                st.markdown(f'''<div class="metric-card">
                  <h3 style="color:{colors[name]}">{name}</h3>
                  <div class="metric-val" style="color:{color}">{status}</div>
                  <div class="metric-lbl">Angle max : {max_th}</div>
                  <div class="metric-lbl">Énergie : {energy}</div>
                </div>''', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# TAB 3 — TUNING
# ════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🔧 Tuning interactif du LQR")
    tc1, tc2 = st.columns(2)
    with tc1:
        q_x   = st.slider("Q[x] position",       0.1,  50.0,   1.0, key="t_qx")
        q_xd  = st.slider("Q[ẋ] vitesse",        0.1,  20.0,   1.0, key="t_qxd")
        q_th  = st.slider("Q[θ] angle",          10.0, 500.0, 100.0, key="t_qth")
        q_thd = st.slider("Q[θ̇] vitesse ang.",   1.0, 100.0,  10.0, key="t_qthd")
        r_t   = st.slider("R coût commande",     0.001,  2.0,   0.1, key="t_r")

    with tc2:
        A, B = get_linear_matrices(PARAMS)
        Q_t  = np.diag([q_x, q_xd, q_th, q_thd])
        R_t  = np.array([[r_t]])
        try:
            P      = solve_continuous_are(A, B, Q_t, R_t)
            K      = np.linalg.inv(R_t) @ B.T @ P
            A_cl   = A - B @ K
            eigs   = np.linalg.eigvals(A_cl)
            stable = all(e.real < 0 for e in eigs)

            st.markdown("#### Gains K")
            for nm, v in zip(["K_x", "K_ẋ", "K_θ", "K_θ̇"], K.flatten()):
                c = "#3fb950" if v < 0 else "#f85149"
                st.markdown(f'<span style="color:{c};font-family:monospace;font-size:1.1rem;font-weight:bold">{nm} = {v:.4f}</span>',
                            unsafe_allow_html=True)

            st.markdown("#### Pôles boucle fermée")
            for e in eigs:
                s = "🟢" if e.real < 0 else "🔴"
                st.markdown(f'<span style="font-family:monospace">{s} {e.real:.3f} {e.imag:+.3f}j</span>',
                            unsafe_allow_html=True)
            if stable: st.success("✅ Système STABLE")
            else:      st.error("❌ Système INSTABLE")
        except Exception as ex:
            st.error(f"Erreur : {ex}")

    if st.button("▶️ Simuler avec ces gains", use_container_width=True):
        h_t = run_simulation("LQR", np.radians(10), 3.0, 20.0,
                             False, [q_x, q_xd, q_th, q_thd], r_t, 10.0)
        st.session_state["hist_tune"] = h_t

    if "hist_tune" in st.session_state:
        h_t  = st.session_state["hist_tune"]
        t_t  = np.array(h_t["t"])
        th_t = np.array(h_t["theta"])
        F_t  = np.array(h_t["F"])
        fig3, ax3 = plt.subplots(2, 1, figsize=(10, 5), facecolor="#0e1117")
        for ax, y, lbl, c in zip(ax3, [th_t, F_t],
                                  ["Angle θ (°)", "Force (N)"],
                                  ["#3fb950", "#d29922"]):
            ax.set_facecolor("#161b22")
            ax.plot(t_t[:len(y)], y, color=c, lw=1.8)
            ax.axhline(0, color="#30363d", lw=0.8, linestyle="--")
            ax.set_ylabel(lbl, color="white", fontsize=9)
            ax.tick_params(colors="white", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#30363d")
            ax.grid(True, alpha=0.15)
        ax3[-1].set_xlabel("Temps (s)", color="white", fontsize=9)
        if h_t["fallen_at"]: st.error(f"❌ Tombé à {h_t['fallen_at']:.2f}s")
        else:                 st.success("✅ STABLE 10s")
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

st.divider()
st.markdown(
    "<center style='color:#8b949e;font-size:0.8rem'>"
    "🛴 Segway Pendule Inversé · Aliou Harber · "
    "<a href='https://github.com/aliouha/Segway_pendule_inverse_control' "
    "style='color:#58a6ff'>GitHub</a></center>",
    unsafe_allow_html=True)