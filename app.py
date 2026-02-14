import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# 1. SETUP & BRANDING
st.set_page_config(page_title="GEXRADAR // QUANT TERMINAL", layout="wide")

# Handle Page State
if 'current_page' not in st.session_state: 
    st.session_state.current_page = "DASHBOARD"

# Get URL params to handle navigation
params = st.query_params
if "page" in params:
    st.session_state.current_page = params["page"]

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;700&display=swap');
        .block-container { padding-top: 1.5rem !important; padding-bottom: 0rem !important; }
        header {visibility: hidden;} 
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0E1117; color: #E0E0E0; }
        
        /* THE HEADER CONTAINER */
        .header-container {
            display: flex;
            align-items: baseline;
            gap: 40px;
            padding-bottom: 15px;
            border-bottom: 1px solid #30363D;
            margin-bottom: 25px;
        }

        /* NEON LOGO FIX */
        .logo-text { 
            font-family: 'JetBrains Mono', monospace; 
            font-size: 26px; font-weight: 700; color: #00C805 !important; 
            letter-spacing: -1px;
            text-decoration: none !important;
            text-shadow: 0 0 10px rgba(0, 200, 5, 0.4);
        }

        /* NAV LINK COLOR FIX (No more blue) */
        .nav-link {
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px;
            font-weight: 400;
            color: #8B949E !important; /* Force original gray */
            text-decoration: none !important;
            text-transform: uppercase;
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .nav-link:hover {
            color: #00C805 !important; /* Neon Green on hover */
            text-shadow: 0 0 15px #00C805 !important;
        }
        
        /* Active page highlight */
        .nav-link-active {
            color: #FFFFFF !important;
            border-bottom: 2px solid #00C805;
        }

        /* CHART BUTTONS (STAY THE SAME) */
        div.stButton > button {
            background-color: #161B22 !important;
            border: 1px solid #30363D !important;
            color: #FFFFFF !important;
            font-family: 'Inter', sans-serif !important;
            border-radius: 4px !important;
            height: 38px !important;
            font-size: 11px !important;
            font-weight: 700 !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
            width: 100% !important;
        }
        div.stButton > button:hover {
            border: 1px solid #00C805 !important;
            color: #00C805 !important;
            box-shadow: 0 0 20px rgba(0, 200, 5, 0.4) !important;
            transform: scale(1.02) !important;
        }

        .sidebar-label { font-size: 10px; color: #666; letter-spacing: 1.5px; font-weight: 800; text-transform: uppercase; margin-top: 20px; border-bottom: 1px solid #333; padding-bottom: 3px; }
        .data-block { background: rgba(255, 255, 255, 0.03); border: 1px solid #30363D; padding: 12px; border-radius: 4px; margin-bottom: 8px; }
        .data-label { font-size: 10px; color: #8B949E; margin-bottom: 2px; }
        .data-value { font-family: 'JetBrains Mono'; font-size: 16px; font-weight: 700; }
        .regime-container { padding: 20px; border-radius: 4px; border: 1px solid #30363D; background: #161B22; margin-bottom: 25px; }
    </style>
    """, unsafe_allow_html=True)

# 2. HEADER
st.markdown(f"""
    <div class="header-container">
        <a href="/?page=DASHBOARD" target="_self" class="logo-text">GEXRADAR<span style="color:white">_</span></a>
        <a href="/?page=DASHBOARD" target="_self" class="nav-link">Dashboard</a>
        <a href="/?page=OPERATIONS" target="_self" class="nav-link">Operations</a>
        <a href="/?page=ABOUT" target="_self" class="nav-link">About</a>
    </div>
""", unsafe_allow_html=True)

# 3. DATA ENGINE (UNCHANGED)
strikes = np.arange(580, 605, 0.5)
spot_price = 592.40
gamma_flip_level = 591.0
momentum_wall = 595.0
max_pain = 590.0
vol_trigger = 588.5

times = pd.date_range(start='9:30', periods=30, freq='10min')
price_walk = 592 + np.cumsum(np.random.normal(0, 0.4, 30))
t_series = pd.DataFrame({'Time': times, 'Price': price_walk, 'GEX': np.random.uniform(-400, 900, 30), 'DEX': np.random.uniform(20, 150, 30), 'CNV': np.random.uniform(-150, 350, 30)})

data = []
for s in strikes:
    is_major = 3.5 if s % 5 == 0 else (1.8 if s % 2.5 == 0 else 0.6)
    call_g = np.exp(-(abs(s - 595)**2)/12) * is_major
    put_g = np.exp(-(abs(s - 588)**2)/12) * is_major
    data.append({'strike': s, 'call_gex': call_g * 6000, 'put_gex': -put_g * 6000, 'vanna': (call_g + put_g) * 0.4, 'oi': int((call_g + put_g) * 60000), 'vol_call': np.random.randint(1500, 9000) * call_g, 'vol_put': np.random.randint(1500, 9000) * put_g, 'vega': (call_g + put_g) * 0.25, 'charm': (call_g - put_g) * 0.12, 'iv': 0.15 + (abs(s - 592)**2 * 0.0008)})
df = pd.DataFrame(data)
is_long_gamma = spot_price > gamma_flip_level
regime_color = "#00C805" if is_long_gamma else "#FF3B3B"

# DASHBOARD PAGE
if st.session_state.current_page == "DASHBOARD":
    # Sidebar (UNCHANGED)
    st.sidebar.markdown("<p class='sidebar-label'>Structural Context</p>", unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="data-block"><div class="data-label">Flow Ratio</div><div class="data-value" style="color:{regime_color}">{df["vol_call"].sum()/(df["vol_call"].sum()+df["vol_put"].sum()):.2f}</div></div><div class="data-block"><div class="data-label">Momentum Wall</div><div class="data-value">${momentum_wall}</div></div><div class="data-block"><div class="data-label">Max Pain</div><div class="data-value">${max_pain}</div></div><div class="data-block"><div class="data-label">Zero Gamma</div><div class="data-value">${gamma_flip_level}</div></div>', unsafe_allow_html=True)
    
    st.sidebar.markdown("<p class='sidebar-label'>Risk & Volatility</p>", unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="data-block"><div class="data-label">Vol Trigger</div><div class="data-value" style="color:#FF3B3B">${vol_trigger}</div></div><div class="data-block"><div class="data-label">Vanna Exposure</div><div class="data-value">${df["vanna"].sum():.2f}M</div></div><div class="data-block"><div class="data-label">Total OI</div><div class="data-value">{df["oi"].sum():,}</div></div>', unsafe_allow_html=True)

    st.sidebar.markdown("<p class='sidebar-label'>Topography Scan</p>", unsafe_allow_html=True)
    topo = df[(df['strike'] > spot_price - 8) & (df['strike'] < spot_price + 8)].sort_values('strike')
    z_topo = np.outer(np.linspace(1, 0.1, 10), (np.abs(topo['call_gex']) + np.abs(topo['put_gex'])).values)
    st.sidebar.plotly_chart(go.Figure(data=[go.Surface(z=z_topo, x=topo['strike'].values, colorscale='Viridis', showscale=False)]).update_layout(height=180, margin=dict(l=0,r=0,b=0,t=0), template="plotly_dark"), use_container_width=True)

    # Main Area
    st.markdown(f'<div class="regime-container"><div style="font-size: 11px; color: #808495; text-transform: uppercase;">Market Regime</div><div style="font-family: \'JetBrains Mono\'; font-size: 24px; font-weight: 700; color: {regime_color};">{"STABLE / LONG GAMMA" if is_long_gamma else "VOLATILE / SHORT GAMMA"}</div></div>', unsafe_allow_html=True)

    # CHART BUTTONS (Glow intact)
    if 'radar_mode' not in st.session_state: st.session_state.radar_mode = "GEX"
    m_cols = st.columns(6)
    modes = ["GEX", "VOL", "HEAT", "SURF", "SMILE", "DELTA"]
    labels = ["OI / GEX", "VOLUME", "HEATMAP", "SURFACE", "SPY SMILE", "NET DELTA"]
    
    for col, mode, label in zip(m_cols, modes, labels):
        with col:
            btn_text = f"● {label}" if st.session_state.radar_mode == mode else label
            if st.button(btn_text, key=f"mode_{mode}"):
                st.session_state.radar_mode = mode
                st.rerun()

    plot_df = df[(df['strike'] > spot_price - 12) & (df['strike'] < spot_price + 12)]
    
    # 3D Logic
    if st.session_state.radar_mode == "SURF":
        days = np.array([1, 7, 30, 60, 90])
        z_vol = np.array([plot_df['iv'].values * (1 + 0.05 * np.log(d)) for d in days])
        fig_main = go.Figure(data=[go.Surface(z=z_vol, x=plot_df['strike'], y=days, colorscale='Thermal')])
    elif st.session_state.radar_mode == "SMILE":
        days = np.array([1, 5, 10, 20, 30])
        z_smile = np.array([0.12 + (abs(plot_df['strike'] - 592)**1.5 * 0.001) / np.sqrt(d) for d in days])
        fig_main = go.Figure(data=[go.Surface(z=z_smile, x=plot_df['strike'], y=days, colorscale='IceFire')])
    elif st.session_state.radar_mode == "DELTA":
        time_steps = np.arange(10)
        z_delta = np.outer(np.sin(time_steps/2), plot_df['call_gex'].values * 0.1)
        fig_main = go.Figure(data=[go.Surface(z=z_delta, x=plot_df['strike'], y=time_steps, colorscale='Portland')])
    elif st.session_state.radar_mode == "HEAT":
        fig_main = go.Figure(data=go.Heatmap(z=np.random.randn(len(plot_df), 20), x=np.arange(20), y=plot_df['strike'], colorscale='Magma'))
    else:
        fig_main = go.Figure()
        if st.session_state.radar_mode == "GEX":
            fig_main.add_trace(go.Bar(y=plot_df['strike'], x=plot_df['call_gex'], orientation='h', marker_color='#00C805'))
            fig_main.add_trace(go.Bar(y=plot_df['strike'], x=plot_df['put_gex'], orientation='h', marker_color='#FF3B3B'))
        else:
            fig_main.add_trace(go.Bar(y=plot_df['strike'], x=plot_df['vol_call'], orientation='h', marker_color='#00C805'))
            fig_main.add_trace(go.Bar(y=plot_df['strike'], x=-plot_df['vol_put'], orientation='h', marker_color='#FF3B3B'))
        
        for lvl, clr, txt, dash in [(gamma_flip_level, "#808495", "FLIP", "dot"), (spot_price, "#FFF", "SPOT", "solid"), (momentum_wall, "#00FFFF", "MOM WALL", "dash"), (max_pain, "#FFD700", "MAX PAIN", "dot"), (vol_trigger, "#FF3B3B", "VOL TRIGGER", "dashdot")]:
            fig_main.add_hline(y=lvl, line_dash=dash, line_color=clr, annotation_text=txt)

    fig_main.update_layout(height=700, template="plotly_dark", showlegend=False, margin=dict(t=10))
    st.plotly_chart(fig_main, use_container_width=True)

    # TRIPLE TS
    st.markdown("### Institutional Time-Series ($MM)")
    for col, color, title in [('GEX', '#00C805', 'GEX $MM'), ('DEX', '#00FFFF', 'DEX $MM'), ('CNV', '#FF00FF', 'CNV $MM')]:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=t_series['Time'], y=t_series[col], line=dict(color=color, width=3)), secondary_y=False)
        fig.add_trace(go.Scatter(x=t_series['Time'], y=t_series['Price'], line=dict(color='white', width=1, dash='dot'), opacity=0.4), secondary_y=True)
        st.plotly_chart(fig.update_layout(height=280, template="plotly_dark", showlegend=False, title=title, margin=dict(t=30, b=20)), use_container_width=True)

    # GREEK MATRIX
    st.markdown("### Greek Sensitivity Matrix")
    v, c = st.columns(2)
    with v: st.plotly_chart(px.bar(plot_df, x='strike', y='vega', title="VEGA", color_discrete_sequence=['#00FFFF']).update_layout(template="plotly_dark", height=300), use_container_width=True)
    with c: st.plotly_chart(px.line(plot_df, x='strike', y='charm', title="CHARM", color_discrete_sequence=['#FF00FF']).update_layout(template="plotly_dark", height=300), use_container_width=True)

elif st.session_state.current_page == "OPERATIONS":
    st.title("OPERATIONS: DEALER MECHANICS")
    st.markdown('<div class="docs-card"><b>FLIP:</b> Level where dealer hedging shifts from stabilizing to destabilizing.</div>', unsafe_allow_html=True)
