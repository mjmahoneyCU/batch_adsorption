import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- CONFIGURATION ---
st.set_page_config(page_title="Batch Adsorption Simulator", layout="wide")
st.title("🧪 Batch Adsorption Simulator")

# --- WIDER IMAGE ---
st.image("BatchExp.png", use_container_width=True)

st.write("Simulates a batch adsorption system using Langmuir kinetics and a material balance model.")

# --- CONTEXTUAL DESCRIPTIONS ---
with st.expander("ℹ️ Simulation Parameter Descriptions", expanded=True):
    st.markdown("""
    Before running your simulation, adjust the sliders below to explore how different system parameters affect batch adsorption performance. Here’s what each parameter means:

    - **Initial Concentration (C₀)**: The starting concentration of protein in the liquid phase (mg/mL). Higher values mean more protein is available for adsorption.

    - **Adsorption Rate Constant (k)**: A lumped kinetic parameter (1/min) that captures how quickly adsorption occurs. It reflects both mass transfer limitations and surface reaction kinetics.

    - **Langmuir Constant (K_Langmuir)**: Defines how tightly the protein binds to the resin. A lower value means tighter binding (higher affinity); a higher value indicates weaker binding.

    - **Maximum Resin Capacity (qmax)**: The maximum amount of protein the resin can hold per mL of resin volume (mg/mL resin). This defines the adsorption saturation limit.

    - **Simulation Time**: The total duration (in minutes) for which the batch process is simulated. Keep in mind that most adsorption processes reach near-equilibrium within 30 minutes.
    """)

# --- FIXED PARAMETERS ---
V_resin = 0.35  # mL
V_solution = 5.0  # mL

# --- SIDEBAR USER INPUT ---
st.sidebar.header("Adjust Simulation Parameters")
c0 = st.sidebar.slider("Initial Concentration (C₀, mg/mL)", 1.0, 50.0, 20.0)
k = st.sidebar.slider("Adsorption Rate Constant (k, 1/min)", 0.01, 5.0, 1.0)
KL = st.sidebar.slider("Langmuir Constant (K-Langmuir, mL/mg)", 0.5, 100.0, 10.0)
qmax = st.sidebar.slider("Maximum Resin Capacity (qmax, mg/mL resin)", 10.0, 200.0, 65.0)
t_end = st.sidebar.slider("Simulation Time (min)", 5, 30, 20)

# --- DEFINE MODEL EQUATIONS ---
def langmuir_odes(t, y):
    C, q = y
    dqdt = k * ((qmax * C) / (KL + C) - q)
    dCdt = - (V_resin / V_solution) * dqdt
    return [dCdt, dqdt]

# --- SOLVE ODEs ---
t_eval = np.linspace(0, t_end, 300)
sol = solve_ivp(langmuir_odes, [0, t_end], [c0, 0.0], t_eval=t_eval, method='BDF')
t, C, q = sol.t, sol.y[0], sol.y[1]

# --- PLOT RESULTS ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t, C, label="Liquid Concentration (mg/mL)", color="blue")
ax.plot(t, q, label="Resin Loading (mg/mL resin)", color="green")
ax.set_xlabel("Time (min)")
ax.set_ylabel("Concentration / Loading")
ax.set_title("Batch Adsorption Simulation (Langmuir Kinetics)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- FINAL VALUES ---
st.subheader("📊 Final State")
st.write(f"**Equilibrium Liquid Concentration**: {C[-1]:.2f} mg/mL")
st.write(f"**Equilibrium Resin Loading**: {q[-1]:.2f} mg/mL resin")

# --- CONCEPTUAL QUESTIONS ---
with st.expander("🧠 Making Sense of the Simulation", expanded=False):
    st.markdown("""
### 1. What factors control how fast the system reaches equilibrium?
**Try this:**  
Set `k = 0.2`, run the simulation. Then increase to `k = 2.0`.  
Keep `C₀ = 20`, `qmax = 65`, `KL = 0.15`.

**What should you observe?**  
- **Answer**: Increasing `k` makes the system reach equilibrium faster, but the final resin loading and final concentration remain the same. The rate constant affects **speed**, not **final state**.

---

### 2. How does initial concentration (C₀) affect the final resin loading and equilibrium concentration?
**Try this:**  
Run simulations with `C₀ = 5`, `20`, and `40`. Keep `qmax`, `k`, and `KL` constant.

**What should you observe?**  
- **Answer**: Higher `C₀` generally leads to higher final resin loading. However, the resin will saturate if `qmax` is too low, so the effect of increasing `C₀` plateaus. The final liquid concentration will also be higher with higher starting amounts.

---

### 3. How does resin capacity (qmax) influence the system’s behavior?
**Try this:**  
Use `qmax = 20`, `65`, and `100` with fixed `C₀ = 20`.

**What should you observe?**  
- **Answer**: Larger `qmax` values allow more protein to bind before saturation. This lowers the final liquid concentration and increases resin loading. However, beyond a certain point, adding more capacity has little effect if all protein is already adsorbed.

---

### 4. What does the Langmuir constant (KL) tell us about binding affinity?
**Try this:**  
Run with `KL = 0.01`, `0.1`, and `0.5`. Keep `C₀ = 20`, `qmax = 65`.

**What should you observe?**  
- **Answer**: Lower `KL` values indicate **tighter binding** (higher affinity). When `KL` is small, the resin binds more strongly even at low concentrations, resulting in lower final liquid concentration. At high `KL`, weaker binding results in less adsorption.

---

### 5. Is adsorption limited by capacity or affinity?
**Try this:**  
- Case 1: `C₀ = 5`, `KL = 0.5`  
- Case 2: `C₀ = 5`, `KL = 0.05`  
Hold other values constant.

**What should you observe?**  
- **Answer**: When C₀ is low, **binding affinity (KL)** matters more than `qmax`. If KL is high (weak binding), little protein adsorbs. Even if `qmax` is large, low affinity limits adsorption.

---

### 6. How does simulation time affect the results?
**Try this:**  
Run with `t_end = 5`, `15`, and `30` minutes for the same parameter set.

**What should you observe?**  
- **Answer**: Shorter simulation time may stop before the system reaches equilibrium. If the curves are still changing at `t_end`, try increasing it. Equilibrium is reached when curves flatten.

---

### 7. What parameter set leads to the most complete removal of protein?
**Challenge:**  
Try to reduce final liquid concentration below `1 mg/mL`.

**What should you observe?**  
- **Answer**: Use **high `qmax`**, **low KL** (tight binding), and a **moderate-to-high k**. Also, avoid starting with a very high C₀. This will maximize adsorption and reduce unbound protein.
""")
