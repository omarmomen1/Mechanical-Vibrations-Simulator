import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- App Configuration ---
st.set_page_config(page_title="Vibration Simulator", layout="wide")
st.title("⚙️ Advanced Mechanical Vibrations Simulator")
st.markdown("Developed to analyze free and forced vibrations of spring-mass-damper systems.")

# --- Sidebar Controls ---
st.sidebar.header("1. System Parameters")
m = st.sidebar.number_input("Mass (m) [kg]", min_value=0.1, value=1.0, step=0.5)
k = st.sidebar.number_input("Stiffness (k) [N/m]", min_value=1.0, value=100.0, step=10.0)
c = st.sidebar.number_input("Damping Coefficient (c) [Ns/m]", min_value=0.0, value=0.0, step=1.0)

st.sidebar.header("2. Initial Conditions")
x0 = st.sidebar.number_input("Initial Displacement (x0) [m]", value=0.1, step=0.05)
v0 = st.sidebar.number_input("Initial Velocity (v0) [m/s]", value=0.0, step=0.1)

st.sidebar.header("3. Harmonic Excitation (Forced Vibration)")
F0 = st.sidebar.number_input("Force Amplitude (F0) [N]", value=0.0, step=1.0)
wf = st.sidebar.number_input("Forcing Frequency (ωf) [rad/s]", value=0.0, step=1.0)

time_duration = st.sidebar.slider("Simulation Time (s)", min_value=2, max_value=20, value=5)

# --- Engineering Calculations ---
wn = np.sqrt(k / m)                 # Natural Frequency
c_critical = 2 * np.sqrt(k * m)     # Critical Damping
zeta = c / c_critical               # Damping Ratio

# Display calculated metrics
col1, col2, col3 = st.columns(3)
col1.metric("Natural Frequency (ωn)", f"{wn:.2f} rad/s")
col2.metric("Critical Damping (cc)", f"{c_critical:.2f} Ns/m")
col3.metric("Damping Ratio (ζ)", f"{zeta:.3f}")

# --- Numerical Simulation (SciPy odeint) ---
# This function defines the differential equation: mx'' + cx' + kx = F0*sin(wf*t)
def vibration_system(y, t, m, c, k, F0, wf):
    x, v = y
    dxdt = v
    dvdt = (F0 * np.sin(wf * t) - c * v - k * x) / m
    return [dxdt, dvdt]

t = np.linspace(0, time_duration, 2000)
y0 = [x0, v0] # Initial state

# Solve the differential equation
solution = odeint(vibration_system, y0, t, args=(m, c, k, F0, wf))
displacement = solution[:, 0]

# --- Plotting ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, displacement, color='#1f77b4', linewidth=2)
ax.axhline(0, color='black', linewidth=1, linestyle='--')
ax.set_title("System Response: Displacement vs. Time")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Displacement x(t) [m]")
ax.grid(True, linestyle=':', alpha=0.7)
ax.set_xlim(0, time_duration)

st.pyplot(fig)

# --- Future Upgrades Section ---
with st.expander("How to handle more complex vibrations"):
    st.write("""
    Because this simulator uses **numerical integration (odeint)** instead of hardcoded formulas, you can easily add complexity in the `vibration_system` function:
    * **Base Excitation:** Change the forcing function to model a car driving over a bumpy road.
    * **Non-linear Springs:** Change `k * x` to `k * x + k3 * x**3`.
    * **Multi-Degree of Freedom (MDOF):** Expand the state vector `y` to include multiple masses and stiffness matrices to simulate a whole building during an earthquake.
    """)