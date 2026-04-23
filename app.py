import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint

# --- App Configuration ---
st.set_page_config(page_title="Ultimate Vibrations Dashboard", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>.stNumberInput > label, .stSlider > label { font-weight: bold; color: #a0aec0; }</style>""", unsafe_allow_html=True)

st.title("⚙️ Comprehensive Mechanical Vibrations Dashboard")
st.markdown("Analyze Free Dynamics, Harmonic Excitation, Resonance, and Transmissibility.")

# --- Sidebar Controls ---
st.sidebar.header("1. System Properties")
m = st.sidebar.number_input("Mass (m) [kg]", min_value=0.1, value=1.0, step=0.1)
k = st.sidebar.number_input("Stiffness (k) [N/m]", min_value=1.0, value=100.0, step=10.0)
c = st.sidebar.slider("Damping (c) [Ns/m]", min_value=0.0, max_value=40.0, value=2.0, step=0.5)

st.sidebar.header("2. Initial Conditions")
x0 = st.sidebar.slider("Initial Displacement [m]", -0.5, 0.5, 0.2, 0.05)
v0 = st.sidebar.slider("Initial Velocity [m/s]", -5.0, 5.0, 0.0, 0.5)

st.sidebar.header("3. Harmonic Excitation (Forced)")
F0 = st.sidebar.slider("Force Amplitude (F0) [N]", 0.0, 50.0, 0.0, 1.0)
wf = st.sidebar.slider("Forcing Frequency (ωf) [rad/s]", 0.0, 30.0, 0.0, 0.5)

time_duration = st.sidebar.slider("Simulation Time (s)", 2, 20, 5)

# --- Core Calculations ---
wn = np.sqrt(k / m)                 
c_critical = 2 * np.sqrt(k * m)     
zeta = c / c_critical               
freq_ratio = wf / wn if wn > 0 else 0

if zeta == 0: color = "#00d2ff"; sys_type = "Undamped"
elif zeta < 1: color = "#00ff88"; sys_type = f"Underdamped (ζ={zeta:.2f})"
elif zeta == 1: color = "#ffaa00"; sys_type = "Critically Damped"
else: color = "#ff4444"; sys_type = f"Overdamped (ζ={zeta:.2f})"

# --- Top Metrics ---
cols = st.columns(5)
cols[0].metric("System State", sys_type)
cols[1].metric("Nat. Freq (ωn)", f"{wn:.2f} rad/s")
cols[2].metric("Damping Ratio (ζ)", f"{zeta:.3f}")
cols[3].metric("Freq Ratio (r=ωf/ωn)", f"{freq_ratio:.2f}")
if freq_ratio > 0:
    mag_factor = 1 / np.sqrt((1 - freq_ratio**2)**2 + (2 * zeta * freq_ratio)**2)
    cols[4].metric("Magnification (M)", f"{mag_factor:.2f}x")
else:
    cols[4].metric("Magnification (M)", "N/A")

# --- Tabs for Complete Syllabus Coverage ---
tab1, tab2, tab3 = st.tabs(["🕒 Time & State Space (Interactive)", "🌊 Resonance Curve (Bode)", "🏗️ Transmissibility"])

# --- TAB 1: Animated Time Domain & Phase Portrait ---
with tab1:
    # Interactive Time Scrubber for the "Moving Graph"
    current_time = st.slider("🕰️ Time Scrubber (Drag to animate tracer dot)", 0.0, float(time_duration), 0.0, 0.01)

    def vibration_system(y, t, m, c, k, F0, wf):
        x, v = y
        return [v, (F0 * np.cos(wf * t) - c * v - k * x) / m]

    t = np.linspace(0, time_duration, 2000)
    solution = odeint(vibration_system, [x0, v0], t, args=(m, c, k, F0, wf))
    displacement, velocity = solution[:, 0], solution[:, 1]

    # Find the exact values at the current scrubbed time
    time_idx = (np.abs(t - current_time)).argmin()
    current_x = displacement[time_idx]
    current_v = velocity[time_idx]

    fig1 = make_subplots(rows=1, cols=2, subplot_titles=("Displacement vs. Time", "Phase Portrait"), horizontal_spacing=0.1)

    # Static Lines
    fig1.add_trace(go.Scatter(x=t, y=displacement, mode='lines', line=dict(color=color, width=2)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=displacement, y=velocity, mode='lines', line=dict(color='#b19cd9', width=2)), row=1, col=2)

    # Moving Tracer Dots based on Scrubber
    fig1.add_trace(go.Scatter(x=[current_time], y=[current_x], mode='markers', marker=dict(color='white', size=12, line=dict(color='red', width=2))), row=1, col=1)
    fig1.add_trace(go.Scatter(x=[current_x], y=[current_v], mode='markers', marker=dict(color='white', size=12, line=dict(color='red', width=2))), row=1, col=2)

    fig1.update_layout(height=450, plot_bgcolor='rgba(15, 15, 15, 1)', paper_bgcolor='rgba(15, 15, 15, 1)', font=dict(color='white'), showlegend=False, margin=dict(t=40, b=20))
    fig1.update_xaxes(gridcolor='#333'); fig1.update_yaxes(gridcolor='#333', zerolinecolor='#666')
    st.plotly_chart(fig1, use_container_width=True)

# --- TAB 2: Resonance Curves (Magnification Factor) ---
with tab2:
    st.markdown("Shows how the amplitude multiplies as forcing frequency approaches natural frequency ($r=1$).")
    r_vals = np.linspace(0, 3, 500)
    
    fig2 = go.Figure()
    # Plot curves for various damping ratios to show the family of curves
    for z_val in [0.05, 0.1, 0.2, 0.5, 1.0]:
        M_vals = 1 / np.sqrt((1 - r_vals**2)**2 + (2 * z_val * r_vals)**2)
        fig2.add_trace(go.Scatter(x=r_vals, y=M_vals, mode='lines', name=f'ζ = {z_val}'))
    
    # Add a marker for the current system state
    if freq_ratio > 0 and freq_ratio <= 3:
        fig2.add_trace(go.Scatter(x=[freq_ratio], y=[mag_factor], mode='markers', name='Current State', marker=dict(color='white', size=12, symbol='star')))

    fig2.update_layout(height=450, title="Magnification Factor vs. Frequency Ratio", xaxis_title="Frequency Ratio (r = ωf/ωn)", yaxis_title="Magnification (M)", plot_bgcolor='rgba(15, 15, 15, 1)', paper_bgcolor='rgba(15, 15, 15, 1)', font=dict(color='white'), yaxis=dict(range=[0, 10]))
    fig2.update_xaxes(gridcolor='#333'); fig2.update_yaxes(gridcolor='#333')
    st.plotly_chart(fig2, use_container_width=True)

# --- TAB 3: Transmissibility ---
with tab3:
    st.markdown("Shows the ratio of the force transmitted to the base/floor versus the applied force. Critical for vibration isolation.")
    fig3 = go.Figure()
    
    for z_val in [0.05, 0.1, 0.2, 0.5, 1.0]:
        TR_vals = np.sqrt(1 + (2 * z_val * r_vals)**2) / np.sqrt((1 - r_vals**2)**2 + (2 * z_val * r_vals)**2)
        fig3.add_trace(go.Scatter(x=r_vals, y=TR_vals, mode='lines', name=f'ζ = {z_val}'))

    fig3.update_layout(height=450, title="Force Transmissibility (TR) vs. Frequency Ratio", xaxis_title="Frequency Ratio (r = ωf/ωn)", yaxis_title="Transmissibility (TR)", plot_bgcolor='rgba(15, 15, 15, 1)', paper_bgcolor='rgba(15, 15, 15, 1)', font=dict(color='white'), yaxis=dict(range=[0, 5]))
    fig3.update_xaxes(gridcolor='#333'); fig3.update_yaxes(gridcolor='#333')
    
    # Draw the isolation zone line (r = sqrt(2))
    fig3.add_vline(x=np.sqrt(2), line_width=2, line_dash="dash", line_color="green", annotation_text="Isolation Region (r > √2)")
    st.plotly_chart(fig3, use_container_width=True)