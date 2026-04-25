import streamlit as st
import numpy as np
import plotly.graph_objects as go
import wave
import struct
import io

# ==========================================
# 1. APP CONFIGURATION & LABVIEW THEME
# ==========================================
st.set_page_config(page_title="VibraLab R&D", page_icon="🎛️", layout="wide")

st.markdown("""
<style>
    /* R&D Lab Instrument Background */
    .stApp {
        background-color: #121212;
        color: #e0e0e0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Hardware Control Knobs (Inputs) */
    div[data-baseweb="input"] > div {
        background-color: #1e1e1e !important;
        border: 2px solid #333333 !important;
        border-radius: 2px !important;
        border-bottom: 2px solid #00ff41 !important; /* Oscilloscope Green Accent */
    }
    
    /* Monospace for all numeric data */
    input[type="number"], .stMarkdown p {
        font-family: 'Consolas', 'Courier New', monospace !important;
        color: #ffffff !important;
    }
    
    /* Instrument Header */
    .lab-header {
        background-color: #1a1a1a;
        padding: 15px 20px;
        border: 1px solid #333;
        border-left: 5px solid #00ff41;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .lab-header h1 {
        color: #e0e0e0;
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .status-light {
        color: #00ff41;
        font-weight: bold;
        animation: blink 2s infinite;
    }
    @keyframes blink { 0% {opacity: 1;} 50% {opacity: 0.4;} 100% {opacity: 1;} }

    /* Telemetry Panels */
    .telemetry-box {
        background-color: #0a0a0a;
        border: 1px solid #333;
        padding: 15px;
        text-align: center;
    }
    .telemetry-val {
        color: #00ff41;
        font-size: 2rem;
        font-weight: bold;
    }
    .telemetry-lbl {
        color: #888;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="lab-header">
    <h1>🎛️ VibraLab Signal Processing & Dynamics</h1>
    <div class="status-light">● DAQ ACTIVE</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. HARDWARE CONTROL MATRIX (TOP)
# ==========================================
st.markdown("### SYSTEM PARAMETERS")
ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns(5)

with ctrl1: m = st.number_input("Mass (kg)", min_value=0.1, value=10.0, step=0.5)
with ctrl2: k = st.number_input("Stiffness (N/m)", min_value=1.0, value=4000.0, step=100.0)
with ctrl3: c = st.number_input("Damping (Ns/m)", min_value=0.0, value=20.0, step=5.0)
with ctrl4: F0 = st.number_input("Force Amplitude (N)", min_value=0.0, value=100.0, step=10.0)
with ctrl5: omega = st.number_input("Force Freq (rad/s)", min_value=0.0, value=20.0, step=1.0)

# ==========================================
# 3. MATHEMATICAL DIAGNOSTICS (TELEMETRY)
# ==========================================
omega_n = np.sqrt(k / m)
freq_hz = omega_n / (2 * np.pi)
zeta = c / (2 * np.sqrt(k * m))

if zeta < 1:
    sys_type = "Underdamped"
    omega_d = omega_n * np.sqrt(1 - zeta**2)
elif zeta == 1:
    sys_type = "Critically Damped"
    omega_d = 0
else:
    sys_type = "Overdamped"
    omega_d = 0

st.markdown("<br>", unsafe_allow_html=True)
tel1, tel2, tel3, tel4 = st.columns(4)
tel1.markdown(f"<div class='telemetry-box'><div class='telemetry-lbl'>Natural Freq (ωn)</div><div class='telemetry-val'>{omega_n:.2f} rad/s</div></div>", unsafe_allow_html=True)
tel2.markdown(f"<div class='telemetry-box'><div class='telemetry-lbl'>Damping Ratio (ζ)</div><div class='telemetry-val'>{zeta:.3f}</div></div>", unsafe_allow_html=True)
tel3.markdown(f"<div class='telemetry-box'><div class='telemetry-lbl'>System Regime</div><div class='telemetry-val' style='color:#ffb703;'>{sys_type}</div></div>", unsafe_allow_html=True)
tel4.markdown(f"<div class='telemetry-box'><div class='telemetry-lbl'>Freq Ratio (ω/ωn)</div><div class='telemetry-val'>{omega/omega_n:.2f}</div></div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 4. CUSTOM RK4 PHYSICS ENGINE (NUMERICAL SOLVER)
# ==========================================
# Initial conditions
x0, v0 = 0.0, 0.0
t_max = 10.0
dt = 0.005
n_steps = int(t_max / dt)

t_arr = np.linspace(0, t_max, n_steps)
x_arr = np.zeros(n_steps)
v_arr = np.zeros(n_steps)

x_arr[0], v_arr[0] = x0, v0

# Fast pure-NumPy 4th Order Runge-Kutta Integrator
for i in range(1, n_steps):
    t = t_arr[i-1]
    x, v = x_arr[i-1], v_arr[i-1]
    
    def derivatives(t_curr, x_curr, v_curr):
        force = F0 * np.sin(omega * t_curr)
        accel = (force - c * v_curr - k * x_curr) / m
        return v_curr, accel

    k1_v, k1_a = derivatives(t, x, v)
    k2_v, k2_a = derivatives(t + dt/2, x + k1_v*dt/2, v + k1_a*dt/2)
    k3_v, k3_a = derivatives(t + dt/2, x + k2_v*dt/2, v + k2_a*dt/2)
    k4_v, k4_a = derivatives(t + dt, x + k3_v*dt, v + k3_a*dt)
    
    x_arr[i] = x + (dt/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    v_arr[i] = v + (dt/6) * (k1_a + 2*k2_a + 2*k3_a + k4_a)

# ==========================================
# 5. OSCILLOSCOPE DISPLAYS (TABS)
# ==========================================
tab1, tab2, tab3 = st.tabs(["📺 Time Oscilloscope", "🌀 Phase Space Orbit", "🔊 Acoustic Synthesizer"])

# Helper function for oscilloscope styling
def apply_scope_style(fig):
    fig.update_layout(
        plot_bgcolor='#0a0a0a', paper_bgcolor='#121212',
        font=dict(color="#00ff41", family="Courier New"),
        xaxis=dict(showgrid=True, gridcolor='#222', zeroline=True, zerolinecolor='#444'),
        yaxis=dict(showgrid=True, gridcolor='#222', zeroline=True, zerolinecolor='#444'),
        margin=dict(l=40, r=20, t=40, b=40)
    )
    return fig

with tab1:
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(x=t_arr, y=x_arr, mode='lines', line=dict(color='#00ff41', width=2), name='Displacement (m)'))
    fig_time.add_trace(go.Scatter(x=t_arr, y=v_arr, mode='lines', line=dict(color='#ff3366', width=1), name='Velocity (m/s)'))
    fig_time.update_layout(title="Transient & Steady-State Response", height=500)
    st.plotly_chart(apply_scope_style(fig_time), use_container_width=True)

with tab2:
    st.markdown("<p style='color:#888;'>Visualizing the energy state orbit (Displacement vs. Velocity).</p>", unsafe_allow_html=True)
    fig_phase = go.Figure()
    fig_phase.add_trace(go.Scatter(x=x_arr, y=v_arr, mode='lines', line=dict(color='#00ff41', width=1.5)))
    fig_phase.update_layout(title="Poincaré Phase Portrait", xaxis_title="Displacement (x)", yaxis_title="Velocity (dx/dt)", height=500)
    st.plotly_chart(apply_scope_style(fig_phase), use_container_width=True)

with tab3:
    st.markdown("### 🔊 Structural Acoustic Signature")
    st.markdown("This tool converts the calculated mechanical vibration waveform directly into an audible pulse. High stiffness ($k > 50000$) will produce higher pitches.")
    
    # 1. Generate high-res waveform for audio (2 seconds at 44.1kHz)
    sample_rate = 44100
    duration = 2.0
    t_audio = np.linspace(0, duration, int(sample_rate * duration))
    
    # Fast analytical decay generation for the sound effect
    audio_wave = np.exp(-zeta * omega_n * t_audio) * np.sin(omega_d * t_audio)
    
    # Normalize to 16-bit integer for standard WAV audio
    audio_norm = np.int16((audio_wave / np.max(np.abs(audio_wave) + 1e-9)) * 32767)
    
    # 2. Encode to raw WAV bytes in memory
    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1) # Mono
        wav_file.setsampwidth(2) # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_norm.tobytes())
    
    # 3. Output to Streamlit audio player
    st.audio(audio_buffer.getvalue(), format="audio/wav")
    
    st.success("✅ Waveform Synthesized. Press Play to hear the structural response.")
