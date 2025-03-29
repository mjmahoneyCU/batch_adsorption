import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.set_page_config(page_title="Batch Adsorption Simulator", layout="centered")
st.title("🧪 Batch Adsorption Simulator")
st.write("Simulates a batch system using a linear driving force (LDF) model and Langmuir isotherm.")

# --- Sidebar parameters ---
st.sidebar.header("Adjust Simulation Parameters")

c0 = st.sidebar.slider("Initial Concentration (c₀, mg/mL)", 1.0, 50.0, 20.0)
Ka = st.sidebar.slider("Adsorption Rate Constant (Ka, 1/min)", 0.01, 5.0, 1.0)
KL = st.sidebar.slider("Langmuir Constant (KL, mL/mg)", 0.001, 1.0, 0.15, step=0.001, format="%.3f")
qmax = st.sidebar.slider("Maximum Resin Capacity (qmax, mg/mL resin)", 10.0, 100.0, 65.0)
eps = st.sidebar.slider("Porosity (ε)", 0.1, 0.9, 0.4)
t_end = st.sidebar.slider("Simulation Time (min)", 10, 300, 100)

# --- ODE model ---
def batch_adsorption(t, y):
    c, q = y
    dqdt = Ka * ((qmax * KL * c) / (1 + KL * c) - q)
    dcdt = - ((1 - eps) / eps) * dqdt
    return [dcdt, dqdt]

# --- Solve ODE ---
t_eval = np.linspace(0, t_end, 300)
sol = solve_ivp(batch_adsorption, [0, t_end], [c0, 0.0], t_eval=t_eval, method='BDF')
t, c, q = sol.t, sol.y[0], sol.y[1]
mass_bound = q * (1 - eps)  # mg/mL column

# --- Plot Results ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t, c, label="Liquid Concentration (mg/mL)", color="blue")
ax.plot(t, q, label="Resin Loading (mg/mL resin)", color="green")
ax.plot(t, mass_bound, label="Mass Adsorbed (mg/mL column)", color="purple")
ax.set_xlabel("Time (min)")
ax.set_ylabel("Concentration / Mass")
ax.set_title("Batch Adsorption Simulation (Langmuir + LDF)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- Final values summary ---
st.subheader("📊 Final State")
st.write(f"**Final Liquid Concentration**: {c[-1]:.2f} mg/mL")
st.write(f"**Final Resin Loading**: {q[-1]:.2f} mg/mL resin")
st.write(f"**Final Mass Bound**: {mass_bound[-1]:.2f} mg/mL column")